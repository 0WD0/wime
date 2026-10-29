#!/usr/bin/env python3
"""Apply recorded source-quality decisions before a sentence dataset is frozen."""
import argparse
from collections import Counter
import json
from pathlib import Path

from replay import fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    args=parser.parse_args()
    if args.out.with_suffix('.dvc').exists() or (args.out/'curation.json').exists():
        raise ValueError('curation applies once, before registering an immutable dataset')
    review=json.loads(args.review.read_text())
    manifest=json.loads((args.out/'manifest.json').read_text())
    before=fingerprint(args.out/'manifest.json')
    counts=Counter({k:v for k,v in manifest['counts'].items() if k.startswith('excluded_')})
    rejected=[]
    for split in ('train','dev'):
        path=args.out/(split+'.jsonl')
        assert fingerprint(path)==manifest['outputs'][split]
        temporary=path.with_suffix('.curated')
        with path.open() as source,temporary.open('w') as target:
            for line in source:
                row=json.loads(line)
                reason=review['documents'].get(row['document'])
                if reason is None and row['category']=='technical' and not row['terminal']:
                    reason='technical target ends at a source line break, not sentence/clause punctuation'
                if reason:
                    rejected.append(dict(id=row['id'],document=row['document'],reason=reason))
                    counts['excluded_quality_review']+=1
                    continue
                target.write(line)
                counts[f'{split}/{row["source"]}']+=1
                counts[f'{split}/length/'+('8-24' if len(row['gold'][0])<=24 else '25-60')]+=1
                counts[f'{split}/unit/{row["unit_type"]}']+=1
                counts[f'boundary/{row["boundaries"][0]["rule"] if row["boundaries"] else "whole-only"}']+=1
        temporary.replace(path)
    curation=dict(input_manifest_sha256=before,review_sha256=fingerprint(args.review),
                  code_sha256=fingerprint(Path(__file__)),rejected=rejected,
                  policy='Retain ordinary unpunctuated user utterances; require actual terminal punctuation for technical prose. Reject reviewed corrupt documents as a group.')
    (args.out/'curation.json').write_text(json.dumps(curation,ensure_ascii=False,indent=2)+'\n')
    manifest.update(counts=dict(counts),curation=dict(path='curation.json',sha256=fingerprint(args.out/'curation.json')),
                    outputs={s:fingerprint(args.out/(s+'.jsonl')) for s in ('train','dev')})
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(rejected=len(rejected),counts=counts),ensure_ascii=False))


if __name__=='__main__':main()
