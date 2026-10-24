#!/usr/bin/env python3
"""Replay actual wime candidates; gold labels are used only after inference.

The handwritten cases are a development smoke set, not a quality benchmark.
Report rescues, regressions and candidate coverage as well as aggregate top-1.
Worker times measure model evaluation; IPC times include serial request/response
and first-use model loading. The interactive UI displays traditional candidates
first and requests neural results asynchronously after its debounce delay.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time


def percentile(values, fraction):
    return round(sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)], 3) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    package = Path(__file__).resolve().parent.parent
    root = package.parents[1]
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--worker', type=Path, default=root / 'bazel-bin/emacs-packages/wime/neural/wime-neural')
    parser.add_argument('--database', type=Path, default=root / 'bazel-bin/emacs-packages/wime/wanxiang.db')
    parser.add_argument('--cases', type=Path, default=package / 'test/neural-cases.json')
    parser.add_argument('--baseline-lisp', type=Path)
    parser.add_argument('--grammar-dir', type=Path)
    parser.add_argument('--grammar-module', type=Path, default=root / 'bazel-bin/emacs-packages/wime/librime-grammar.so')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def record(name, width, lisp):
        path = args.output / f'{name}.jsonl'
        env = dict(os.environ, WIME_DATABASE=str(args.database.resolve()),
                   WIME_CASES=str(args.cases.resolve()), WIME_RECORDING=str(path.resolve()),
                   WIME_WIDTH=str(width))
        if args.grammar_dir:
            env.update(WIME_GRAMMAR=str(args.grammar_dir.resolve()),
                       WIME_GRAMMAR_MODULE=str(args.grammar_module.resolve()))
        subprocess.run(['emacs', '-Q', '--batch', '-L', str(lisp),
                        '-l', str(lisp / 'wime.el'), '-l',
                        str(package / 'test/record-neural.el')], env=env, check=True)
        return [json.loads(line) for line in path.read_text().splitlines()]

    baseline = record('baseline', 1, args.baseline_lisp or package)
    beam = record('beam', 8, package)
    before = {r['case']: r for r in baseline if r['final']}
    model_times, ipc_times, decisions, final = [], [], [], []
    worker = subprocess.Popen([str(args.worker.resolve()), str(args.model.resolve())],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        for row in beam:
            candidates = row['candidates']
            eligible = (candidates and not candidates[0]['dictionary']
                        and candidates[0]['end'] == row['syllables']
                        and sum(c['end'] == row['syllables'] for c in candidates) >= 2)
            chosen = row['leader']
            if eligible:
                # The model receives no gold, timing, case labels, or target length.
                request = {k: row[k] for k in ('id', 'context', 'syllables', 'candidates')}
                started = time.perf_counter()
                worker.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
                worker.stdin.flush()
                reply = json.loads(worker.stdout.readline())
                elapsed = (time.perf_counter() - started) * 1000
                assert reply['id'] == row['id']
                index = reply['promoted']
                if index is not None:
                    assert candidates[index]['end'] == row['syllables']
                    chosen = candidates[index]['text']
                model_times.append(reply['elapsed_ms'])
                ipc_times.append(elapsed)
                decisions.append(dict(reply, ipc_ms=elapsed, chosen=chosen))
            if row['final']:
                gold = row['gold']
                final.append(dict(case=row['case'], gold=gold, baseline=before[row['case']]['leader'],
                                  beam=row['leader'], neural=chosen,
                                  covered=any(c['text'] in gold and c['end'] == row['syllables']
                                              for c in candidates[:9])))
    finally:
        worker.stdin.close()
        worker.wait(timeout=5)
    summary = dict(cases=len(final), keystrokes=len(beam), model_requests=len(model_times),
                   model=str(args.model.resolve()), worker=str(args.worker.resolve()),
                   database=str(args.database.resolve()), grammar=bool(args.grammar_dir),
                   execution='Emacs source, serial per-keystroke replay; UI debounce excluded',
                   correct={name: sum(r[name] in r['gold'] for r in final)
                            for name in ('baseline', 'beam', 'neural')},
                   covered=sum(r['covered'] for r in final),
                   rescued=sum(r['beam'] not in r['gold'] and r['neural'] in r['gold'] for r in final),
                   regressed=sum(r['beam'] in r['gold'] and r['neural'] not in r['gold'] for r in final),
                   search_ms={name: {'p50': percentile([r['search_ms'] for r in rows], .5),
                                    'p95': percentile([r['search_ms'] for r in rows], .95)}
                              for name, rows in [('baseline', baseline), ('beam', beam)]},
                   model_ms={'p50': percentile(model_times, .5), 'p95': percentile(model_times, .95)},
                   response_ms={'p50': percentile(ipc_times, .5), 'p95': percentile(ipc_times, .95)},
                   note='Handwritten development cases; no statistically supported quality claim.')
    for name, value in [('summary', summary), ('results', final), ('decisions', decisions)]:
        (args.output / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
