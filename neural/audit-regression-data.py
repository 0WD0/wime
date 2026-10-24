#!/usr/bin/env python3
from pathlib import Path
from collections import Counter,deque
from difflib import SequenceMatcher
import json,hashlib
import argparse
parser=argparse.ArgumentParser(description="Audit training coverage of regression contrasts after prediction; never a training sampler.")
parser.add_argument("--cases",type=Path,required=True)
parser.add_argument("--root",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
r=args.root;cases=json.loads(args.cases.read_text());patterns=set()
for row in cases:
 a,b=row['gold'][0],row['after'];spans=[]
 for op,i,j,k,l in SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
  if op=='equal':continue
  # Count both contextual contrasts and multi-character changed spans.
  pair=[a[max(0,i-1):min(len(a),j+1)],b[max(0,k-1):min(len(b),l+1)]]
  spans.append(pair);patterns.update(pair)
  raw=[a[i:j],b[k:l]]
  if min(map(len,raw))>=2 and raw not in spans:spans.append(raw);patterns.update(raw)
 row['contrasts']=spans
patterns=sorted(patterns);edges=[{}];failure=[0];outputs=[set()]
for i,word in enumerate(patterns):
 state=0
 for char in word:
  if char not in edges[state]:edges[state][char]=len(edges);edges.append({});failure.append(0);outputs.append(set())
  state=edges[state][char]
 outputs[state].add(i)
queue=deque(edges[0].values())
while queue:
 state=queue.popleft()
 for char,nxt in edges[state].items():
  queue.append(nxt);f=failure[state]
  while f and char not in edges[f]:f=failure[f]
  failure[nxt]=edges[f].get(char,0);outputs[nxt].update(outputs[failure[nxt]])
def matches(text):
 state=0;seen=set()
 for char in text:
  while state and char not in edges[state]:state=failure[state]
  state=edges[state].get(char,0);seen.update(outputs[state])
 return seen
counts={};metadata={}
for name,relative in [('original_chat','data/train.jsonl'),('original_technical','programming/train.jsonl'),('replay','replay/train.jsonl'),('new','diverse/train.jsonl')]:
 path=r/relative;c=Counter();total=0;chars=0
 with path.open() as f:
  for line in f:
   x=json.loads(line);text=x['gold'][0];c.update(matches(text));total+=1;chars+=len(text)
 counts[name]={patterns[i]:v for i,v in c.items()};
 with path.open('rb') as source: digest=hashlib.file_digest(source,'sha256').hexdigest()
 metadata[name]=dict(rows=total,characters=chars,sha256=digest)
 print(name,total,flush=True)
for row in cases:
 row['contrast_counts']=[dict(gold=a,prediction=b,counts={name:dict(gold=c.get(a,0),prediction=c.get(b,0)) for name,c in counts.items()}) for a,b in row.pop('contrasts')]
 for key in ('candidates',):row.pop(key)
summary=dict(sources=metadata,note='Post-hoc target-row occurrence counts of multi-character differing spans and contextual spans with one character on each side; raw prepared rows before augmentation. Frequency is evidence of coverage, not proof of causation.',cases=cases)
args.out.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
