#!/usr/bin/env python3
"""Write the Japanese reading table for a rescorer's tokenizer (see japanese.py)."""
import argparse
import json
from pathlib import Path

from japanese import build_table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rescorer", required=True, help="transformers model directory (its tokenizer)")
    ap.add_argument("--jmdict", default="/usr/share/tagainijisho/jmdict.db")
    ap.add_argument("--kanjidic", default="/usr/share/tagainijisho/kanjidic2.db")
    ap.add_argument("--lexicon", type=Path, help="build-japanese-lexicon.py's output")
    ap.add_argument("--top", type=int, default=4, help="lexicon writings kept per reading")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.rescorer)
    units = build_table(tokenizer, args.jmdict, args.kanjidic,
                        json.loads(args.lexicon.read_text()) if args.lexicon else None, args.top)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(rescorer=Path(args.rescorer).name, units=units), ensure_ascii=False))
    print(len(units), "readings,", sum(map(len, units.values())), "units")


if __name__ == "__main__":
    main()
