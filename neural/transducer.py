"""Aligned syllable transducer: one output character per typed syllable.

Three parts. A causal character network (the CharLM transformer, initialized
from a conditional checkpoint) reads the left context, an OUTPUT marker and
the characters so far; it never sees keys, so its states survive keystrokes.
A small bidirectional encoder reads every syllable of the composition as
initial + final + tone embeddings, each possibly unknown: toneless, toned and
half-typed input share one model. Optionally the top cross_layers blocks of
the character network also attend to the encoder; the blocks below stay
key-independent. A joint head combines the character state before position i
with the syllable state at i and scores characters through the tied
character embedding.
"""
import dataclasses
import json
import math

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from conditional import load_model
from composition import OUTPUT
from model import CharLM, Config
import syllables

FORMAT = "wime-syllable-transducer-v1"
CONTEXT = 64


@dataclasses.dataclass
class EncoderConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    positions: int = 64
    # Without the marker the context runs straight into the composition.
    marker: bool = True
    # Top character-network blocks followed by cross-attention to the encoder.
    cross_layers: int = 0


class SyllableEncoder(nn.Module):
    def __init__(self, cfg, out):
        super().__init__()
        self.initial = nn.Embedding(len(syllables.INITIALS), cfg.n_embd)
        self.final = nn.Embedding(len(syllables.FINALS), cfg.n_embd)
        self.tone = nn.Embedding(syllables.TONES, cfg.n_embd)
        self.pos = nn.Embedding(cfg.positions, cfg.n_embd)
        layer = nn.TransformerEncoderLayer(cfg.n_embd, cfg.n_head, 4 * cfg.n_embd, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layer, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.n_embd)
        self.out = nn.Linear(cfg.n_embd, out)

    def forward(self, features, padding):
        """FEATURES [B, S, 3] of (initial, final, tone); PADDING [B, S] is True past the end."""
        x = (self.initial(features[..., 0]) + self.final(features[..., 1]) + self.tone(features[..., 2])
             + self.pos(torch.arange(features.shape[1], device=features.device)))
        return self.out(self.norm(self.blocks(x, src_key_padding_mask=padding)))


class CrossAttention(nn.Module):
    """Residual attention from character states to syllable states; starts as identity."""

    def __init__(self, n, heads):
        super().__init__()
        self.norm = nn.LayerNorm(n)
        self.attention = nn.MultiheadAttention(n, heads, batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, x, memory, padding):
        return x + self.attention(self.norm(x), memory, memory, key_padding_mask=padding, need_weights=False)[0]


class Transducer(nn.Module):
    def __init__(self, predictor_cfg, encoder_cfg):
        super().__init__()
        self.predictor = CharLM(predictor_cfg)
        self.encoder = SyllableEncoder(encoder_cfg, predictor_cfg.n_embd)
        n = predictor_cfg.n_embd
        self.from_predictor = nn.Linear(n, n)
        self.from_encoder = nn.Linear(n, n, bias=False)
        self.joint = nn.Linear(n, n)
        self.cross = nn.ModuleList(CrossAttention(n, predictor_cfg.n_head) for _ in range(encoder_cfg.cross_layers))
        self.cfg, self.encoder_cfg = predictor_cfg, encoder_cfg

    def states(self, ids, encoded=None, padding=None):
        """Character-network states [B, T, C] for token IDS [B, T]; the top
        blocks attend to ENCODED [B, S, C] (PADDING [B, S]) when coupled."""
        p = self.predictor
        x = p.tok(ids) + p.pos(torch.arange(ids.shape[1], device=ids.device))
        first = len(p.blocks) - len(self.cross)
        for i, block in enumerate(p.blocks):
            x = block(x)
            if i >= first:
                x = self.cross[i - first](x, encoded, padding)
        return p.ln_f(x)

    def logits(self, states, encoded):
        """Joint scores over characters for aligned STATES and ENCODED, both [..., C]."""
        z = self.joint(F.gelu(self.from_predictor(states) + self.from_encoder(encoded)))
        return z @ self.predictor.tok.weight.T


def from_conditional(path, extra_tokens=(), encoder_cfg=EncoderConfig()):
    """A transducer whose character network starts from a conditional
    checkpoint, its vocabulary extended as train-conditional.py does."""
    base, vocab, metadata = load_model(path, extend=True, extra_tokens=extra_tokens)
    model = Transducer(base.cfg, encoder_cfg)
    model.predictor.load_state_dict(base.state_dict())
    return model, vocab


def save(model, vocab, path, **metadata):
    state = {k: v.detach().to(torch.float16).contiguous() for k, v in model.state_dict().items()
             if k != "predictor.head.weight"}
    save_file(state, str(path), metadata=dict(
        {k: str(v) for k, v in metadata.items()}, format=FORMAT, vocab=json.dumps(vocab, ensure_ascii=False),
        config=json.dumps(dataclasses.asdict(model.cfg)), encoder=json.dumps(dataclasses.asdict(model.encoder_cfg))))


def load(path):
    with safe_open(str(path), "pt") as f:
        metadata = f.metadata()
        if metadata.get("format") != FORMAT:
            raise ValueError(f"expected a {FORMAT} model")
        state = {k: f.get_tensor(k).float() for k in f.keys()}
    model = Transducer(Config(**json.loads(metadata["config"])), EncoderConfig(**json.loads(metadata["encoder"])))
    state["predictor.head.weight"] = state["predictor.tok.weight"]
    model.load_state_dict(state)
    return model, json.loads(metadata["vocab"]), metadata


def sequence(index, context, text, marker=True):
    """Character-network ids: BOS, the last context characters, OUTPUT unless
    MARKER is false, TEXT. The state before composition character i sits at
    offset + i."""
    room = min(CONTEXT, 127 - marker - len(text))
    context = "".join(c for c in context if not 0xE000 <= ord(c) <= 0xF8FF)[-room:] if room > 0 else ""
    ids = [2] + [index.get(c, 1) for c in context] + [index[OUTPUT]] * marker + [index.get(c, 1) for c in text]
    return ids, len(context) + marker


class Evaluator:
    """The fields the decoder reads from an evaluator, for a trained transducer."""

    def __init__(self, path, device="cuda"):
        self.model, self.vocab, self.metadata = load(path)
        self.model.to(device).eval()
        self.index = {c: i for i, c in enumerate(self.vocab)}
        self.device = device

    def evaluate(self, *_, **__):
        # Verification needs a conditional model; the editor keeps its lattice.
        raise ValueError("the transducer only decodes")


class Prompt:
    """Left context and syllable features; the encoder runs once per prompt."""

    def __init__(self, context, features, marker):
        self.context, self.features, self.marker, self.encoded = context, features, marker, None

    def __len__(self):
        # BOS, context, OUTPUT: the decoder adds characters against the window.
        return 1 + self.marker + min(CONTEXT, len(self.context))


def decoder_class(base):
    """BASE (decoder.Decoder) with its conditional-model methods replaced."""

    class TransducerDecoder(base):
        def model_prompt(self, context, code, tones=()):
            if len(code) % 2:
                raise ValueError("expected complete syllables")
            return Prompt(context, syllables.typed(code, tones),
                          self.evaluator.model.encoder_cfg.marker)

        def encoded(self, prompt):
            if prompt.encoded is None:
                features = torch.tensor([prompt.features], device=self.device)
                prompt.encoded = self.evaluator.model.encoder(
                    features, torch.zeros(features.shape[:2], dtype=torch.bool, device=self.device))[0]
            return prompt.encoded

        def joint(self, prompts, texts, slots):
            """Log probabilities at (text, slot) pairs SLOTS: [len(slots), V]."""
            model, index = self.evaluator.model, self.evaluator.index
            rows = [sequence(index, p.context, t, p.marker) for p, t in zip(prompts, texts)]
            width = max(len(ids) for ids, _ in rows)
            ids = torch.tensor([ids + [0] * (width - len(ids)) for ids, _ in rows], device=self.device)
            # Prompts are few (one per segment); gather their slots in one index.
            unique = list({id(p): p for p in prompts}.values())
            which = {id(p): k for k, p in enumerate(unique)}
            table = torch.nn.utils.rnn.pad_sequence([self.encoded(p) for p in unique], batch_first=True)
            owner = torch.tensor([which[id(p)] for p in prompts], device=self.device)
            lengths = torch.tensor([len(p.features) for p in unique], device=self.device)
            padding = torch.arange(table.shape[1], device=self.device)[None] >= lengths[:, None]
            states = model.states(ids, table[owner], padding[owner])
            picked = states[[i for i, _ in slots], [rows[i][1] + j for i, j in slots]]
            encoded = table[[which[id(prompts[i])] for i, _ in slots], [j for _, j in slots]]
            return model.logits(picked, encoded).float().log_softmax(-1)

        @torch.inference_mode()
        def model_next_each(self, prompts, texts):
            return self.joint(prompts, texts, [(i, len(t)) for i, t in enumerate(texts)])

        @torch.inference_mode()
        def model_score_each(self, prompts, texts):
            slots = [(i, j) for i, t in enumerate(texts) for j in range(len(t))]
            if not slots:
                return [0.0] * len(texts)
            values = self.joint(prompts, texts, slots)
            targets = torch.tensor([self.evaluator.index.get(texts[i][j], 1) for i, j in slots], device=self.device)
            picked = values[torch.arange(len(slots), device=self.device), targets]
            rows = torch.tensor([i for i, _ in slots], device=self.device)
            return torch.zeros(len(texts), device=self.device).index_add_(0, rows, picked).tolist()

    return TransducerDecoder
