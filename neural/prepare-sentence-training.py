#!/usr/bin/env python3
"""Create a protected, source-matched sentence cohort for representation/sampling controls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from dataset_registry import Registry
from replay import Protection, fingerprint
from sentence_sampling import boundary, legacy_span, LENGTHS


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def protections(registry):
    references = registry.entry('mandarin-context-v2-r3-parquet')['exclude'] + [
        dict(dataset=name, files=['cases']) for name in ('sentence-trajectories-dev-v1', 'sentence-trajectories-test-v1')]
    paths = []
    for ref in references:
        registry.verify(ref['dataset'], ref['files'])
        paths.extend(registry.path(ref['dataset'], name) for name in ref['files'])
    return Protection(list(dict.fromkeys(paths)))


def select(args):
    import jieba_fast.posseg
    registry = Registry()
    registry.verify('cleaned-multisource-v4', ['manifest', 'documents'])
    protection = protections(registry)
    # The new test's stricter contiguous overlap rule applies even when a
    # different training document contains the same sentence or its context.
    test = json.loads(registry.path('sentence-trajectories-test-v1', 'cases').read_text())
    fragments = defaultdict(set)
    for row in test:
        text = row['gold'][0]
        size = min(12, len(text))
        fragments[size].update(text[i:i+size] for i in range(len(text)-size+1))
    args.out.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(registry.path('cleaned-multisource-v4', 'documents').as_uri()+'?mode=ro&immutable=1', uri=True)
    db.row_factory = sqlite3.Row
    selected, counts, seen_groups = [], Counter(), set()
    for split, per_kind in (('train', (args.per_source//5, args.per_source*4//5)), ('dev', (20, 80))):
        for source in ('wildchat', 'forum', 'zhihu', 'fineweb', 'zhwiki'):
            buckets = [[], []]
            # Content-hash IDs give a deterministic order independent of source
            # file order. One retained sentence per source group caps repetition.
            for doc in db.execute('SELECT * FROM documents WHERE split=? AND source=? ORDER BY id', (split,source)):
                if doc['group_id'] in seen_groups or doc['id'] in protection.documents or doc['group_id'] in protection.groups:
                    continue
                choices = []
                for match in re.finditer(r'[^。！？!?\n]+[。！？!?\n]?', doc['text']):
                    raw = match.group(); text = raw.rstrip('。！？!?\n').strip()
                    if (not 8 <= len(text) <= 24 or not re.fullmatch(r'[\u3400-\u4dbf\u4e00-\u9fff]+', text)
                            or len(set(text)) < 4 or re.search(r'(的话|但是|而且|[把被与或但不就才该])$',text)):
                        continue
                    start = match.start()+len(raw)-len(raw.lstrip())
                    terminal = raw[-1] if raw[-1] in '。！？!?' else ''
                    if doc['text'][start:start+len(text)+len(terminal)] != text+terminal:
                        continue
                    context = doc['text'][max(0,start-64):start]
                    row = dict(document=doc['id'],group=doc['group_id'],source=source,category=doc['category'],
                               content=doc['content'],split=split,offset=start,context=context,gold=[text],terminal=terminal)
                    if protection.contains(row):
                        counts['excluded_protected'] += 1; continue
                    combined = context+text
                    if any(any(combined[i:i+n] in keys for i in range(len(combined)-n+1)) for n,keys in fragments.items()):
                        counts['excluded_test_text_overlap'] += 1; continue
                    segments = jieba_fast.posseg.lcut(text)
                    ends = boundary(text, segments)
                    kind = bool(ends)
                    if len(buckets[kind]) >= per_kind[kind]:
                        continue
                    identity = hashlib.sha256(f'sentence-training-v1:{doc["id"]}:{start}'.encode()).hexdigest()[:24]
                    choices.append(dict(row,id=identity,boundaries=ends))
                if choices:
                    row = min(choices, key=lambda r:r['id'])
                    buckets[bool(row['boundaries'])].append(row)
                    seen_groups.add(doc['group_id'])
                if all(len(bucket) >= count for bucket,count in zip(buckets,per_kind)):
                    break
            # Preserve the paired/whole-only ratio when a source exhausts its
            # valid groups. Never fabricate a boundary to fill the quota.
            paired = min(len(buckets[1]), 4*len(buckets[0]))
            whole = paired//4
            kept = (buckets[0][:whole] + buckets[1][:4*whole]) if split=='train' else buckets[0]+buckets[1]
            selected += kept
            counts[f'{split}/{source}/whole_only'] = sum(not r['boundaries'] for r in kept)
            counts[f'{split}/{source}/paired'] = sum(bool(r['boundaries']) for r in kept)
            print(json.dumps(dict(split=split,source=source,selected=len(kept),available=list(map(len,buckets)))),flush=True)
    db.close()
    with (args.out/'proposals.jsonl').open('w') as stream:
        for row in selected:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    dump(args.out/'selection.json', dict(counts=dict(counts),protected=protection.files,
        source=registry.snapshot('cleaned-multisource-v4',['manifest','documents']),
        sentence_length=[8,24],selection='Hash-ordered documents; one sentence per group; 80% rule-supported boundaries, 20% whole-only within each source.',
        limitations='Rule-derived training boundaries, not individually human reviewed or measured user commit frequencies.',
        proposals_sha256=fingerprint(args.out/'proposals.jsonl')))


def annotate(args):
    from corpus_readings import Lexicon, Reader
    from mandarin_frontend import MandarinFrontend
    from corpus_tones import reading_notes, validate_tones, READING_SCHEMA, TONE_POLICY
    from corpus_integrity import reading_assets
    from trajectories import validate
    selection = json.loads((args.out/'selection.json').read_text())
    assert selection['proposals_sha256'] == fingerprint(args.out/'proposals.jsonl')
    proposals = [json.loads(line) for line in (args.out/'proposals.jsonl').read_text().splitlines()]
    lexicon = Lexicon(args.database)
    reader = Reader(lexicon, MandarinFrontend(args.g2pw))
    counts, records = Counter(), []
    for i in range(0,len(proposals),32):
        batch = proposals[i:i+32]
        readings = reader.readings([r['context']+r['gold'][0]+r['terminal'] for r in batch])
        for row, reading in zip(batch, readings, strict=True):
            start=len(row['context']); end=start+len(row['gold'][0])
            if not all(reading['keys'][start:end]):
                counts['excluded_unreachable_reading']+=1;continue
            row.update(input=''.join(reading['keys'][start:end]),pinyin=reading['pinyin'][start:end],tones=reading['tones'][start:end],
                       reading_window=[max(0,row['offset']-64),row['offset']+len(row['gold'][0])+len(row['terminal'])],
                       legacy_span=legacy_span(row['gold'][0],lexicon.segment(row['gold'][0]),row['id']))
            notes = reading_notes(reading,start,end)
            if notes:row['reading_notes']=notes
            validate(row);validate_tones(row)
            records.append(row)
            counts[f'{row["split"]}/{row["source"]}']+=1
            counts[f'boundary/{row["boundaries"][0]["rule"] if row["boundaries"] else "whole-only"}']+=1
        if i%320==0:print(json.dumps(dict(annotated=min(i+32,len(proposals)),total=len(proposals))),flush=True)
    for split in ('train','dev'):
        with (args.out/(split+'.jsonl')).open('x') as stream:
            for row in records:
                if row['split']==split:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    dump(args.out/'manifest.json',dict(format='wime-sentence-training-v1',counts=dict(counts),selection=selection,
        reading_schema=READING_SCHEMA,tone_policy=TONE_POLICY,
        window='actual preceding 64 characters + full source sentence + original terminal; derive states from these same readings',
        legacy_control=dict(lengths=LENGTHS,policy='One deterministic word-aligned span from each same source sentence; original length lottery, no original multi-million-row population.'),
        state_weights=dict(whole=.6,prefix=.2,continuation=.2),
        training_boundaries='Deterministic conservative rules, sample-reviewed; not individually human verified.',
        reading_assets=reading_assets(args.g2pw,args.g2pw/'chinese-roberta-wwm-ext-large'),
        database_sha256=fingerprint(args.database),
        outputs={s:fingerprint(args.out/(s+'.jsonl')) for s in ('train','dev')},
        code={f:fingerprint(Path(__file__).with_name(f)) for f in ('prepare-sentence-training.py','sentence_sampling.py','mandarin_frontend.py','corpus_readings.py','corpus_tones.py')}))
    print(json.dumps(dict(counts),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('select','annotate'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--per-source',type=int,default=2500)
    parser.add_argument('--database',type=Path)
    parser.add_argument('--g2pw',type=Path)
    args=parser.parse_args()
    (select if args.operation=='select' else annotate)(args)
