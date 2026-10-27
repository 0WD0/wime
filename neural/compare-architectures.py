#!/usr/bin/env python3
"""Frozen small-sample ablation of candidate generation and final ranking.

The legacy decoder, a tokenizer-independent small-model character beam, and
their union define three candidate pools. Compare joint canonical-token-path
LM ranking with the existing weighted fusion on exactly the same pools.
References are consulted only after pools and predictions have been fixed.
This is an offline prototype, not an interactive worker implementation.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    settings = next(m for m in plan["models"] if m["label"] == args.model)
    output = args.plan.parent / args.model
    output.mkdir(exist_ok=True)
    if any((output / (name + ".jsonl")).exists() for name in plan["suites"]):
        raise SystemExit("Refusing to overwrite an existing experiment")
    sys.path.insert(0, plan["code"])
    import torch
    from decoder import Decoder, Units, readings
    from inference import ConditionalEvaluator
    from rescorer import Rescorer
    source = Path(__file__).with_name("character-proposals.py")
    spec = importlib.util.spec_from_file_location("character_proposals", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with (Path.home() / ".local/share/wime/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evaluator = ConditionalEvaluator(plan["conditional"], device="cuda")
        lm = Rescorer(settings["path"], settings["weight"])
        table = readings(plan["database"])
        decoder = Decoder(evaluator, lm, Units(lm.tokenizer, table))
        proposer = module.CharacterProposals(
            evaluator, table, beam=plan["parameters"]["character_beam"],
            per_hypothesis=plan["parameters"]["character_per_hypothesis"])
        decoder.decode("", "nihk", anchor=2)

        @torch.inference_mode()
        def joint_scores(context, texts):
            # Canonical paths, not a sum over all tokenizations. All candidate
            # sequences share exactly the same excluded prefix probability.
            context = context or "\n"
            sequences = [lm.tokenizer(context + t)["input_ids"] for t in texts]
            common = 0
            for values in zip(*sequences):
                if len(set(values)) != 1:
                    break
                common += 1
            inserted_seed = not common
            if inserted_seed:
                seed = lm.tokenizer("\n", add_special_tokens=False)["input_ids"]
                sequences = [seed + s for s in sequences]
                common = len(seed)
            if len(texts) == 1:
                # Ranking is trivial, but retain a meaningful nonempty tail.
                common = min(common, len(sequences[0]) - 1)
            values = []
            for start in range(0, len(sequences), 8):
                batch = sequences[start:start + 8]
                inputs = torch.full((len(batch), max(map(len, batch))), lm.pad,
                                    dtype=torch.long, device=decoder.device)
                for i, seq in enumerate(batch):
                    inputs[i, :len(seq)] = torch.tensor(seq, device=decoder.device)
                hidden = lm.model.model(input_ids=inputs).last_hidden_state
                owners, positions, targets = [], [], []
                for i, seq in enumerate(batch):
                    for j in range(common, len(seq)):
                        owners.append(i), positions.append(j - 1), targets.append(seq[j])
                sums = torch.zeros(len(batch), device=decoder.device)
                for offset in range(0, len(targets), 128):
                    owner = torch.tensor(owners[offset:offset + 128], device=decoder.device)
                    position = torch.tensor(positions[offset:offset + 128], device=decoder.device)
                    target = torch.tensor(targets[offset:offset + 128], device=decoder.device)
                    logits = lm.model.lm_head(hidden[owner, position]).float().log_softmax(-1)
                    picked = logits.gather(1, target[:, None]).squeeze(1)
                    sums.index_add_(0, owner, picked)
                values.extend(sums.tolist())
            return [q / len(t) for q, t in zip(values, texts)], dict(
                common_prefix_tokens=common, inserted_newline_seed=inserted_seed)

        summaries = {}
        for name, suite in plan["suites"].items():
            assert digest(suite["path"]) == suite["sha256"]
            rows = json.loads(Path(suite["path"]).read_text())
            results = []
            with (output / (name + ".jsonl")).open("x") as stream:
                for row in rows:
                    context, code = row["context"][-64:], row["input"]
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    decoder.session = None
                    legacy = decoder.decode(context, code, anchor=2)
                    torch.cuda.synchronize()
                    after_legacy = time.perf_counter()
                    characters = proposer.decode(context, code, anchor=2)
                    torch.cuda.synchronize()
                    after_characters = time.perf_counter()
                    pools = dict(legacy=[t for t, _ in legacy], character=[t for t, _ in characters])
                    texts = list(dict.fromkeys(pools["legacy"] + pools["character"]))
                    assert texts and all(len(t) * 2 == len(code) for t in texts)
                    assert all(all(code[2*i:2*i+2] in table.get(c, {}) for i, c in enumerate(t)) for t in texts)
                    prompt, _ = decoder.prompts(context, code, 2, None)
                    m = []
                    for offset in range(0, len(texts), 32):
                        chunk = texts[offset:offset + 32]
                        m.extend(s / len(t) for s, t in zip(decoder.model_score(prompt, chunk), chunk))
                    q, tokenization = joint_scores(context, texts)
                    scores = {t: dict(conditional=a, joint_lm=b, fused=a + settings["weight"] * b)
                              for t, a, b in zip(texts, m, q)}
                    def best(pool, field):
                        return max(pool, key=lambda t: scores[t][field]) if pool else ""
                    predictions = dict(
                        legacy=legacy[0][0] if legacy else "",
                        legacy_pool_joint_lm=best(pools["legacy"], "joint_lm"),
                        legacy_pool_joint_fusion=best(pools["legacy"], "fused"),
                        character_only=characters[0][0] if characters else "",
                        character_pool_joint_lm=best(pools["character"], "joint_lm"),
                        union_pool_joint_lm=best(texts, "joint_lm"),
                        union_pool_joint_fusion=best(texts, "fused"))
                    torch.cuda.synchronize()
                    ended = time.perf_counter()
                    # Labels enter only after all generation and ranking.
                    result = dict(id=row["id"], document=row["document"], category=row["category"],
                                  variant=row["variant"], context=context, input=code, gold=row["gold"],
                                  predictions=predictions, correct={k: v in row["gold"] for k, v in predictions.items()},
                                  oracle={k: any(t in row["gold"] for t in pool)
                                          for k, pool in dict(pools, union=texts).items()},
                                  candidates=scores, pools=pools, tokenization=tokenization,
                                  gold_outside_conditional_vocabulary=[g for g in row["gold"]
                                      if any(evaluator.index.get(c, 1) == 1 for c in g)],
                                  elapsed_ms=dict(legacy=1000*(after_legacy-started),
                                      character_generation=1000*(after_characters-after_legacy),
                                      union_rescoring=1000*(ended-after_characters)))
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    results.append(result)
                    if len(results) % 15 == 0 or len(results) == len(rows):
                        print(json.dumps(dict(model=args.model, suite=name, completed=len(results),
                            total=len(rows), top1={k: sum(r["correct"][k] for r in results) for k in predictions})), flush=True)
            summaries[name] = dict(rows=len(results),
                top1={k: sum(r["correct"][k] for r in results) for k in results[0]["predictions"]},
                oracle={k: sum(r["oracle"][k] for r in results) for k in results[0]["oracle"]},
                seed_rows=sum(r["tokenization"]["inserted_newline_seed"] for r in results))
        (output / "summary.json").write_text(json.dumps(dict(
            model=settings, summaries=summaries, plan_sha256=digest(args.plan),
            script_sha256=digest(__file__), character_proposals_sha256=digest(source),
            conditional_sha256=digest(plan["conditional"])), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
