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
    import shutil
    import time
    from corpus_readings import Lexicon, Reader
    from mandarin_frontend import MandarinFrontend
    from corpus_tones import reading_notes, validate_tones, READING_SCHEMA, TONE_POLICY
    from corpus_integrity import reading_assets
    from trajectories import validate
    selection=json.loads((args.out/'selection.json').read_text())
    assert selection['proposals_sha256']==fingerprint(args.out/'proposals.jsonl')
    assert all(fingerprint(Path(p))==h for p,h in selection['protected'].items())
    code={f:fingerprint(Path(__file__).with_name(f)) for f in
          ('prepare-sentence-training.py','sentence_sampling.py','mandarin_frontend.py','corpus_readings.py','corpus_tones.py')}
    assets=reading_assets(args.g2pw,args.g2pw/'chinese-roberta-wwm-ext-large')
    database_sha256=fingerprint(args.database)
    recipe=dict(proposals=selection['proposals_sha256'],code=code,reading_assets=assets,database_sha256=database_sha256)
    cache=args.out/'.annotation'
    cache.mkdir(exist_ok=True)
    if (cache/'recipe.json').exists():
        if json.loads((cache/'recipe.json').read_text())!=recipe:
            raise ValueError('annotation recipe changed; use a new output directory')
    else:dump(cache/'recipe.json',recipe)
    proposals=[json.loads(line) for line in (args.out/'proposals.jsonl').read_text().splitlines()]
    protection=Protection([Path(p) for p in selection['protected']])
    lexicon=Lexicon(args.database)
    reader=Reader(lexicon,MandarinFrontend(args.g2pw))
    print(json.dumps(dict(providers=reader.frontend.providers,total=len(proposals))),flush=True)
    started=time.monotonic()
    for part_start in range(0,len(proposals),512):
        part=cache/f'{part_start:08d}.json'
        if part.exists():continue
        output=[]
        for i in range(part_start,min(part_start+512,len(proposals)),32):
            batch=proposals[i:min(i+32,part_start+512)]
            readings=reader.readings([r['context']+r['gold'][0]+r['terminal'] for r in batch])
            for proposal,reading in zip(batch,readings,strict=True):
                row=dict(proposal);start=len(row['context']);end=start+len(row['gold'][0])
                if not all(reading['keys'][start:end]):
                    output.append(dict(id=row['id'],reason='unreachable_reading'));continue
                row.update(input=''.join(reading['keys'][start:end]),pinyin=reading['pinyin'][start:end],tones=reading['tones'][start:end],
                    reading_window=[max(0,row['offset']-64),row['offset']+len(row['gold'][0])+len(row['terminal'])],
                    legacy_span=legacy_span(row['gold'][0],lexicon.segment(row['gold'][0]),row['id']))
                notes=reading_notes(reading,start,end)
                if notes:row['reading_notes']=notes
                validate(row);validate_tones(row)
                text=row['gold'][0]
                cuts=[(0,len(text)),tuple(row['legacy_span'])]
                for point in row['boundaries']:cuts.extend(((0,point['end']),(point['end'],len(text))))
                if any(protection.contains(dict(row,context=(row['context']+text[:a])[-64:],
                        input=row['input'][2*a:2*b],gold=[text[a:b]])) for a,b in cuts):
                    output.append(dict(id=row['id'],reason='protected_derived_state'));continue
                output.append(dict(row=row))
        dump(part.with_suffix('.partial'),output)
        part.with_suffix('.partial').replace(part)
        print(json.dumps(dict(annotated=min(part_start+512,len(proposals)),total=len(proposals),seconds=time.monotonic()-started)),flush=True)
    counts=Counter();rejected=[]
    streams={split:(args.out/(split+'.jsonl')).open('w') for split in ('train','dev')}
    try:
        for part_start in range(0,len(proposals),512):
            for result in json.loads((cache/f'{part_start:08d}.json').read_text()):
                if 'reason' in result:
                    rejected.append(result);counts['excluded_'+result['reason']]+=1;continue
                row=result['row'];streams[row['split']].write(json.dumps(row,ensure_ascii=False)+'\n')
                counts[f'{row["split"]}/{row["source"]}']+=1
                counts[f'{row["split"]}/length/'+('8-24' if len(row['gold'][0])<=24 else '25-60')]+=1
                counts[f'{row["split"]}/unit/{row["unit_type"]}']+=1
                counts[f'boundary/{row["boundaries"][0]["rule"] if row["boundaries"] else "whole-only"}']+=1
    finally:
        for stream in streams.values():stream.close()
    dump(args.out/'rejected.json',rejected)
    dump(args.out/'manifest.json',dict(format='wime-sentence-training-v2',counts=dict(counts),selection=selection,
        reading_schema=READING_SCHEMA,tone_policy=TONE_POLICY,
        window='actual preceding 64 characters + full source sentence or punctuation-delimited clause + original terminal; slice all states from this contextual reading',
        legacy_control=dict(lengths=LENGTHS,policy='One deterministic word-aligned span per source unit; not an independent training row.'),
        state_weights=dict(whole=.6,prefix=.2,continuation=.2),
        training_boundaries='Conservative rules; no boundary quota or synthetic cut. Actual possible state shares recorded by the trainer.',
        reading_assets=assets,database_sha256=database_sha256,
        outputs={split:fingerprint(args.out/(split+'.jsonl')) for split in ('train','dev')},code=code))
    shutil.rmtree(cache)
    print(json.dumps(dict(counts),ensure_ascii=False),flush=True)


def validate_release(args):
    from corpus_tones import validate_tones
    from trajectories import validate
    manifest=json.loads((args.out/'manifest.json').read_text())
    selection=manifest['selection']
    assert fingerprint(args.out/'documents.sqlite3')==selection['documents_sha256']
    assert all(fingerprint(Path(p))==h for p,h in selection['protected'].items())
    protection=Protection([Path(p) for p in selection['protected']])
    db=sqlite3.connect((args.out/'documents.sqlite3').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    db.row_factory=sqlite3.Row
    dictionary=sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    legal=set(dictionary.execute('SELECT word,code FROM entry WHERE length(word)=1 AND length(code)=2'))
    groups={};identities=set();targets=set();counts=Counter();group_bands=Counter();documents=set()
    for split in ('train','dev'):
        assert fingerprint(args.out/(split+'.jsonl'))==manifest['outputs'][split]
        with (args.out/(split+'.jsonl')).open() as stream:
            for line in stream:
                row=json.loads(line);validate(row);validate_tones(row)
                doc=db.execute('SELECT * FROM documents WHERE id=?',(row['document'],)).fetchone()
                text=row['gold'][0];offset=row['offset']
                assert row['split']==doc['split']==split and row['group']==doc['group_id']
                assert row['source']==doc['source'] and row['category']==doc['category']
                assert groups.setdefault(row['group'],split)==split
                assert row['id'] not in identities and text not in targets
                identities.add(row['id']);targets.add(text);documents.add(row['document'])
                assert doc['text'][offset:offset+len(text)+len(row['terminal'])]==text+row['terminal']
                assert doc['text'][max(0,offset-64):offset]==row['context']
                assert hashlib.sha256(doc['text'].encode()).hexdigest()==row['content']==doc['content']
                assert row['reading_window']==[max(0,offset-64),offset+len(text)+len(row['terminal'])]
                assert all((char,row['input'][2*i:2*i+2]) in legal for i,char in enumerate(text))
                cuts=[(0,len(text)),tuple(row['legacy_span'])]
                for point in row['boundaries']:cuts.extend(((0,point['end']),(point['end'],len(text))))
                for a,b in cuts:
                    assert 0<=a<b<=len(text)
                    assert not protection.contains(dict(row,context=(row['context']+text[:a])[-64:],input=row['input'][2*a:2*b],gold=[text[a:b]]))
                    counts['derived_states_checked']+=1
                group_bands[(row['group'],len(text)>24)]+=1
                assert group_bands[(row['group'],len(text)>24)]<=selection['selection']['technical_per_band' if row['category']=='technical' else 'per_group_band']
                counts[split]+=1;counts[split+'/paired' if row['boundaries'] else split+'/whole_only']+=1
    result=dict(counts=dict(counts),groups={s:sum(v==s for v in groups.values()) for s in ('train','dev')},documents=len(documents),
        source_offsets_contexts_and_hashes=True,train_dev_groups_and_targets_disjoint=True,
        contextual_readings_structurally_valid=True,dictionary_reachable=True,protected_sources_and_states_absent=True,
        limits='Rule boundaries and tone values are not individually human verified or audio-backed.')
    dump(args.out/'validation.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('select','annotate','validate'))
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
    {'select':select,'annotate':annotate,'validate':validate_release}[args.operation](args)
