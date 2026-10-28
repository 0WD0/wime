"""One pinned Mandarin frontend: contextual G2PW, phrase corrections, sandhi.

The imported GPT-SoVITS frontend routes characters deterministically between
its model and dictionaries. Its corrected readings feed the tone rules; there
is no independent second annotation or vote. Original text is never rewritten.
"""

import ast
from functools import lru_cache
import importlib
import importlib.machinery
from pathlib import Path
import re
import sys
import types


HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
TEXT_ROOT = Path(__file__).resolve().parents[3] / "third_party/gpt-sovits/GPT_SoVITS/text"
NAMESPACE = "_wime_gpt_sovits_text"


def numbered_reading(value):
    """Preserve only explicit numbered syllables; missing tones are unknown."""
    if not isinstance(value, str):
        return None
    value = value.lower().replace("u:", "v").replace("ü", "v")
    return value if re.fullmatch(r"[a-z]+[1-5]", value) else None


def _package(name, directory):
    """Register a package path without executing its heavy __init__.py."""
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(directory)]
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = module.__path__
    sys.modules[name] = module


@lru_cache(maxsize=1)
def imported_modules():
    _package(NAMESPACE, TEXT_ROOT)
    _package(NAMESPACE + ".g2pw", TEXT_ROOT / "g2pw")
    _package(NAMESPACE + ".zh_normalization", TEXT_ROOT / "zh_normalization")
    api = importlib.import_module(NAMESPACE + ".g2pw.onnx_api")
    tone = importlib.import_module(NAMESPACE + ".tone_sandhi")
    return api, tone


def validate_literal_config(path):
    """Reject executable statements before upstream imports the pinned config."""
    values = {}
    for statement in ast.parse(Path(path).read_text(), filename=str(path)).body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            raise ValueError(f"model config must contain only literal assignments: {path}")
        name = statement.targets[0].id
        if name.startswith("_"):
            raise ValueError(f"invalid model config attribute {name}")
        try:
            values[name] = ast.literal_eval(statement.value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"nonliteral model config attribute {name}") from error
    return values


@lru_cache(maxsize=1)
def load_corrections():
    """Read the upstream text dictionaries without eval or pickle caches."""
    corrections = {}
    for name in ("polyphonic.rep", "polyphonic-fix.rep"):
        path = TEXT_ROOT / "g2pw" / name
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            word, literal = line.split(":", 1)
            word, values = word.strip(), ast.literal_eval(literal.strip())
            if (not isinstance(values, list) or len(values) != len(word)
                    or any(numbered_reading(value) is None for value in values)):
                raise ValueError(f"invalid pronunciation entry {path}:{line_number}")
            corrections[word] = tuple(numbered_reading(value) for value in values)
    return corrections


class MandarinFrontend:
    """Return final, aligned readings from the imported Mandarin frontend."""

    def __init__(self, model_root=None, *, converter=None, segmenter=None, tone_modifier=None, corrections=None):
        if converter is None:
            root = Path(model_root)
            model = root / "model/G2PWModel"
            tokenizer = root / "chinese-roberta-wwm-ext-large"
            for path in (model / "config.py", model / "POLYPHONIC_CHARS.txt", model / "MONOPHONIC_CHARS.txt",
                         tokenizer / "config.json", tokenizer / "tokenizer_config.json", tokenizer / "vocab.txt"):
                if not path.is_file():
                    raise FileNotFoundError(path)
            if not any((model / name).is_file() for name in ("g2pW.onnx", "g2pw.onnx")):
                raise FileNotFoundError(model / "g2pW.onnx")
            validate_literal_config(model / "config.py")
            api, _ = imported_modules()
            converter = api.G2PWOnnxConverter(model_dir=str(model), model_source=str(tokenizer), style="pinyin",
                                             enable_non_tradional_chinese=True)
            # Freeze behavior independently of the user's shell environment.
            converter.polyphonic_context_chars = 0
            converter.enable_sentence_dedup = True
        if segmenter is None:
            import jieba_fast.posseg
            segmenter = jieba_fast.posseg.lcut
        if tone_modifier is None:
            _, tone = imported_modules()
            tone_modifier = tone.ToneSandhi()
        self.converter = converter
        self.segmenter = segmenter
        self.tone_modifier = tone_modifier
        self.corrections = load_corrections() if corrections is None else corrections

    @property
    def session_g2pw(self):
        return self.converter.session_g2pw

    def _correct(self, word, readings):
        exact = self.corrections.get(word)
        if exact is not None:
            return list(exact), [[f"phrase_correction:{word}"] for _ in word]
        values, rules = list(readings), [[] for _ in word]
        for i, char in enumerate(word):
            replacement = self.corrections.get(char)
            if replacement is not None:
                values[i] = replacement[0]
                rules[i].append(f"character_correction:{char}")
        return values, rules

    def readings(self, sentences):
        # The converter and tone rules see complete sentence windows. Positions
        # remain indices in the original text, including Latin and punctuation.
        predictions = self.converter(sentences)
        if not isinstance(predictions, (list, tuple)) or len(predictions) != len(sentences):
            raise ValueError("Mandarin frontend must return one row per sentence")
        results = []
        for sentence, prediction in zip(sentences, predictions):
            if not isinstance(prediction, (list, tuple)) or len(prediction) != len(sentence):
                raise ValueError("Mandarin frontend predictions must align with every original character")
            raw = [numbered_reading(value) if HAN.fullmatch(char) else None
                   for char, value in zip(sentence, prediction)]
            before, final = [None] * len(sentence), [None] * len(sentence)
            sources = ["unresolved" if HAN.fullmatch(char) else "non_han" for char in sentence]
            rules = [[] for _ in sentence]
            # Unsupported characters are explicit boundaries, never guessed
            # from their neighbours or deleted from an aligned sentence.
            i = 0
            while i < len(sentence):
                if raw[i] is None:
                    i += 1
                    continue
                end = i + 1
                while end < len(sentence) and raw[end] is not None:
                    end += 1
                text = sentence[i:end]
                segments = [(word, pos) for word, pos in self.segmenter(text)]
                if "".join(word for word, _ in segments) != text:
                    raise ValueError("Mandarin segmentation changed original text")
                segments = self.tone_modifier.pre_merge_for_modify(segments, readings=raw[i:end])
                if "".join(word for word, _ in segments) != text:
                    raise ValueError("Mandarin tone grouping changed original text")
                offset = i
                for word, pos in segments:
                    stop = offset + len(word)
                    corrected, word_rules = self._correct(word, raw[offset:stop])
                    if len(corrected) != len(word) or any(numbered_reading(value) is None for value in corrected):
                        raise ValueError("phrase correction must preserve numbered character readings")
                    modified = self.tone_modifier.modified_tone(word, pos, list(corrected))
                    if (len(modified) != len(word) or any(numbered_reading(value) is None for value in modified)
                            or any(a[:-1] != b[:-1] for a, b in zip(corrected, modified))):
                        raise ValueError("tone sandhi must preserve syllables and character alignment")
                    before[offset:stop], final[offset:stop] = corrected, modified
                    for k, (old, new) in enumerate(zip(corrected, modified), offset):
                        applied = word_rules[k - offset]
                        if old != new:
                            applied.append("contextual_tone_sandhi")
                        rules[k] = applied
                        sources[k] = ("contextual_tone_sandhi" if old != new else
                                      "phrase_correction" if applied else "mandarin_frontend")
                    offset = stop
                i = end
            results.append(dict(tone_sources=sources, reading_rules=rules,
                                reading_evidence=dict(frontend_before_sandhi=before, final=final)))
        return results
