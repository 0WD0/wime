#!/usr/bin/env python3
"""Convert a zenz gguf (azooKey's GPT-2 kana-kanji model) into a transformers directory.

The worker then needs only transformers and tokenizers. Reading the gguf
needs llama.cpp's gguf-py (PYTHONPATH=<llama.cpp>/gguf-py), which also
imports sentencepiece. zenz-v3.2-small
(huggingface.co/Miwa-Keita/zenz-v3.2-small-gguf) is Apache-2.0.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

PARTS = {"attn_norm": "ln_1", "ffn_norm": "ln_2", "attn_qkv": "attn.c_attn", "attn_output": "attn.c_proj",
         "ffn_up": "mlp.c_fc", "ffn_down": "mlp.c_proj"}
NAMES = {"token_embd.weight": "transformer.wte.weight", "position_embd.weight": "transformer.wpe.weight",
         "output_norm.weight": "transformer.ln_f.weight", "output_norm.bias": "transformer.ln_f.bias",
         "output.weight": "lm_head.weight"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel

    r = GGUFReader(args.gguf)
    number = lambda k: int(r.fields[k].parts[r.fields[k].data[0]][0])
    strings = lambda k: [bytes(r.fields[k].parts[i]).decode() for i in r.fields[k].data]
    tokens, merges = strings("tokenizer.ggml.tokens"), strings("tokenizer.ggml.merges")
    weights = {}
    for t in r.tensors:
        w = torch.from_numpy(np.array(dequantize(t.data, t.tensor_type), dtype=np.float32)
                             .reshape([int(x) for x in reversed(t.shape)]))
        if t.name in NAMES:
            weights[NAMES[t.name]] = w
            continue
        _, i, part, kind = t.name.split(".")
        name = PARTS[part]
        # GPT-2's Conv1D stores weights input-major.
        weights[f"transformer.h.{i}.{name}.{kind}"] = w.T.contiguous() if kind == "weight" and "." in name else w
    weights.setdefault("lm_head.weight", weights["transformer.wte.weight"])
    eos = tokens.index("</s>")  # the gguf metadata names <s>
    config = GPT2Config(vocab_size=len(tokens), n_positions=number("gpt2.context_length"),
                        n_embd=number("gpt2.embedding_length"), n_layer=number("gpt2.block_count"),
                        n_head=number("gpt2.attention.head_count"), n_inner=number("gpt2.feed_forward_length"),
                        activation_function="gelu_new", bos_token_id=eos, eos_token_id=eos)
    model = GPT2LMHeadModel(config)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or any(".attn.bias" not in m and "masked_bias" not in m for m in missing):
        raise ValueError(f"unexpected tensors: {unexpected} {missing}")
    model.save_pretrained(args.out)
    tokenizer = Tokenizer(models.BPE(vocab={t: i for i, t in enumerate(tokens)},
                                     merges=[tuple(m.split(" ")) for m in merges], unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(args.out / "tokenizer.json"))
    print(f"{args.out}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, eos {eos}")


if __name__ == "__main__":
    main()
