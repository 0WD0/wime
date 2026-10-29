#!/usr/bin/env python3
"""Build a broad, protected sentence/commit corpus with contextual readings."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from dataset_registry import Registry
from replay import Protection, fingerprint
from sentence_sampling import boundary, legacy_span, LENGTHS, source_units, retain_group


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def protections(registry):
    references = registry.entry('mandarin-context-v2-r3-parquet')['exclude'] + [
        dict(dataset=name, files=['cases']) for name in ('sentence-trajectories-dev-v1', 'sentence-trajectories-test-v1')] + [dict(dataset='programming-v1',files=['dev','test'])]
    paths = []
    for ref in references:
        registry.verify(ref['dataset'], ref['files'])
        paths.extend(registry.path(ref['dataset'], name) for name in ref['files'])
    return Protection(list(dict.fromkeys(paths)))


def source_documents(registry, args):
    """Original source splits and IDs survive extraction into this new dataset."""
    db=sqlite3.connect(registry.path('cleaned-multisource-v4','documents').as_uri()+'?mode=ro&immutable=1',uri=True)
    db.row_factory=sqlite3.Row
    for row in db.execute("SELECT * FROM documents WHERE split IN ('train','dev') ORDER BY split,id"):
        doc=dict(row)
        doc.update(origin='cleaned-multisource-v4',source_document=doc['id'],cleaning='[]')
        yield doc
    db.close()
    from technical_prose import documents
    manifest=json.loads(registry.path('programming-v1','manifest').read_text())
    for source,path,metadata in (('python',args.python_archive,manifest),
                                 ('kubernetes',args.kubernetes_archive,manifest['appended_corpus'])):
        if fingerprint(path)!=metadata['archive_sha256']:
            raise ValueError(f'changed pinned archive: {path}')
        for name,raw,text in documents(path,source,metadata['revision']):
            content=hashlib.sha256(text.encode()).hexdigest()
            bucket=int(content[:8],16)%100
            if bucket<10:continue
            split='dev' if bucket<20 else 'train'
            yield dict(id=content,group_id=hashlib.sha256((source+':'+name).encode()).hexdigest(),
                source=source,category='technical',split=split,content=content,
                raw_content=hashlib.sha256(raw.encode()).hexdigest(),text=text,
                origin=metadata['source']+'@'+metadata['revision'],source_document=name,
                cleaning=json.dumps(['technical_prose.render; original legacy document identity retained']))


def select(args):
    import jieba_fast.posseg
    registry=Registry()
    registry.verify('cleaned-multisource-v4',['manifest','documents'])
    registry.verify('programming-v1',['manifest'])
    protection=protections(registry)
    test=json.loads(registry.path('sentence-trajectories-test-v1','cases').read_text())
    fragments=defaultdict(set)
    for row in test:
        text=row['gold'][0];size=min(12,len(text))
        fragments[size].update(text[i:i+size] for i in range(len(text)-size+1))
    args.out.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(args.out/'documents.sqlite3')
    fields=('id','group_id','source','category','split','content','raw_content','text','origin','source_document','cleaning')
    db.execute('CREATE TABLE documents('+','.join(f+' TEXT'+(' PRIMARY KEY' if f=='id' else '') for f in fields)+')')
    buckets=defaultdict(dict);counts=Counter();seen_documents=set()
    for number,doc in enumerate(source_documents(registry,args)):
        if doc['id'] in seen_documents:
            counts['excluded_duplicate_document']+=1;continue
        seen_documents.add(doc['id'])
        if doc['id'] in protection.documents or doc['group_id'] in protection.groups:
            counts['excluded_protected_document']+=1;continue
        found=False
        for item in source_units(doc['text'],args.min_length,args.max_length):
            start=item['offset'];text=item['text'];context=doc['text'][max(0,start-64):start]
            identity=hashlib.sha256(f'sentence-training-v2:{doc["id"]}:{start}'.encode()).hexdigest()[:24]
            row=dict(id=identity,document=doc['id'],group=doc['group_id'],source=doc['source'],
                     category=doc['category'],content=doc['content'],split=doc['split'],offset=start,
                     context=context,gold=[text],terminal=item['terminal'],unit_type=item['unit_type'])
            if protection.contains(row):
                counts['excluded_protected_pair']+=1;continue
            combined=context+text
            if any(any(combined[i:i+n] in keys for i in range(len(combined)-n+1)) for n,keys in fragments.items()):
                counts['excluded_test_text_overlap']+=1;continue
            limit=args.technical_per_band if doc['category']=='technical' else args.per_group_band
            retain_group(buckets[(doc['split'],doc['source'])],row,limit)
            counts[f'candidates/{doc["split"]}/{doc["source"]}']+=1
            found=True
        if found:
            db.execute('INSERT OR IGNORE INTO documents VALUES('+','.join('?' for _ in fields)+')',[doc[f] for f in fields])
        if number%50000==0:
            db.commit();print(json.dumps(dict(documents_scanned=number,candidates=sum(v for k,v in counts.items() if k.startswith('candidates/')))),flush=True)
    seen=set();selected=[]
    # Dev wins exact target duplicates, avoiding train/dev phrase contamination.
    for split,source in sorted(buckets):
        candidates=sorted((row for group in buckets[(split,source)].values() for row in group),key=lambda row:row['id'])
        kept=0
        for row in candidates:
            text=row['gold'][0]
            if text in seen:
                counts['excluded_duplicate_target']+=1;continue
            if args.per_source and kept>=args.per_source:break
            seen.add(text);kept+=1
            row['boundaries']=boundary(text,jieba_fast.posseg.lcut(text))
            selected.append(row)
            counts[f'{split}/{source}/'+('paired' if row['boundaries'] else 'whole_only')]+=1
        print(json.dumps(dict(split=split,source=source,selected=kept)),flush=True)
    db.execute('CREATE TEMP TABLE retained(id TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO retained VALUES(?)',[(n,) for n in {row['document'] for row in selected}])
    db.execute('DELETE FROM documents WHERE id NOT IN (SELECT id FROM retained)')
    db.commit();db.execute('VACUUM');db.close()
    with (args.out/'proposals.jsonl').open('w') as stream:
        for row in selected:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    dump(args.out/'selection.json',dict(counts=dict(counts),protected=protection.files,
        source=registry.snapshot('cleaned-multisource-v4',['manifest','documents']),
        technical=registry.snapshot('programming-v1',['manifest']),
        archives={str(p):fingerprint(p) for p in (args.python_archive,args.kubernetes_archive)},
        sentence_length=[args.min_length,args.max_length],
        selection=dict(order='hash within original split/source group and length band; dev exact targets protected from train',
                       per_group_band=args.per_group_band,technical_per_band=args.technical_per_band,
                       per_source=args.per_source,boundary_quota=None),
        limitations='Source sentences or existing punctuation-delimited clauses; Chinese-only targets. Rule boundaries are not individual human gold.',
        documents_sha256=fingerprint(args.out/'documents.sqlite3'),proposals_sha256=fingerprint(args.out/'proposals.jsonl')))


def annotate(args):
    from corpus_readings import Lexicon, Reader
    from mandarin_frontend import MandarinFrontend
    from corpus_tones import reading_notes, validate_tones, READING_SCHEMA, TONE_POLICY
    from corpus_integrity import reading_assets
    from trajectories import validate
    selection = json.loads((args.out/'selection.json').read_text())
    assert selection['proposals_sha256'] == fingerprint(args.out/'proposals.jsonl')
    proposals = [json.loads(line) for line in (args.out/'proposals.jsonl').read_text().splitlines()]
    protection = Protection([Path(p) for p in selection['protected']])
    lexicon = Lexicon(args.database)
    reader = Reader(lexicon, MandarinFrontend(args.g2pw))
    counts, records, rejected = Counter(), [], []
    for i in range(0,len(proposals),32):
        batch = proposals[i:i+32]
        readings = reader.readings([r['context']+r['gold'][0]+r['terminal'] for r in batch])
        for row, reading in zip(batch, readings, strict=True):
            start=len(row['context']); end=start+len(row['gold'][0])
            if not all(reading['keys'][start:end]):
                counts['excluded_unreachable_reading']+=1
                rejected.append(dict(id=row['id'],reason='unreachable_reading'))
                continue
            row.update(input=''.join(reading['keys'][start:end]),pinyin=reading['pinyin'][start:end],tones=reading['tones'][start:end],
                       reading_window=[max(0,row['offset']-64),row['offset']+len(row['gold'][0])+len(row['terminal'])],
                       legacy_span=legacy_span(row['gold'][0],lexicon.segment(row['gold'][0]),row['id']))
            notes = reading_notes(reading,start,end)
            if notes:row['reading_notes']=notes
            validate(row);validate_tones(row)
            text=row['gold'][0]
            cuts=[(0,len(text)),tuple(row['legacy_span'])]
            for b in row['boundaries']:
                cuts.extend(((0,b['end']),(b['end'],len(text))))
            if any(protection.contains(dict(row,context=(row['context']+text[:a])[-64:],
                        input=row['input'][2*a:2*b],gold=[text[a:b]])) for a,b in cuts):
                counts['excluded_protected_state']+=1
                rejected.append(dict(id=row['id'],reason='protected_derived_state'))
                continue
            records.append(row)
            counts[f'{row["split"]}/{row["source"]}']+=1
            counts[f'boundary/{row["boundaries"][0]["rule"] if row["boundaries"] else "whole-only"}']+=1
        if i%320==0:print(json.dumps(dict(annotated=min(i+32,len(proposals)),total=len(proposals))),flush=True)
    for split in ('train','dev'):
        with (args.out/(split+'.jsonl')).open('x') as stream:
            for row in records:
                if row['split']==split:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    dump(args.out/'rejected.json',rejected)
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
    parser.add_argument('--per-source',type=int,default=0,help='optional smoke-test cap; 0 retains all eligible groups')
    parser.add_argument('--per-group-band',type=int,default=1)
    parser.add_argument('--technical-per-band',type=int,default=12)
    parser.add_argument('--min-length',type=int,default=8)
    parser.add_argument('--max-length',type=int,default=60)
    parser.add_argument('--python-archive',type=Path,default=Path.home()/'.cache/wime-conditional/python-docs-zh-cn.tar.gz')
    parser.add_argument('--kubernetes-archive',type=Path,default=Path.home()/'.cache/wime-conditional/kubernetes-website.tar.gz')
    parser.add_argument('--database',type=Path)
    parser.add_argument('--g2pw',type=Path)
    args=parser.parse_args()
    (select if args.operation=='select' else annotate)(args)
