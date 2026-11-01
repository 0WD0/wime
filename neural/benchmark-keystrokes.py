#!/usr/bin/env python3
"""Per-keystroke decode latency, typing each case key by key.

Each case keeps one decoder session across its keystrokes, as the editor
does; every prefix with a complete first syllable is decoded, although the
editor's debounce skips some. Times include device synchronization. Accuracy
is the top result after the last key equal to the gold text.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import time

import torch

from composition import decode as decode_composition
from decoder import Decoder, Units
from fusion import accelerator
from replay import fingerprint
from rescorer import Rescorer
from transducer import Evaluator

DATA = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "wime"
QWEN = "Qwen/Qwen3.5-2B-Base@b1485b2fa6dfa1287294f269f5fb618e03d52d7c"


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def fetch(spec, directory, file="", **kwargs):
    """SPEC as a local path, else REPO[@REVISION] downloaded from Hugging Face
    into DIRECTORY of the wime data directory, where the README puts it."""
    if Path(spec).exists():
        return Path(spec)
    from huggingface_hub import snapshot_download
    repo, _, revision = spec.partition("@")
    return Path(snapshot_download(repo, revision=revision or None, local_dir=DATA / directory,
                                  allow_patterns=file or None, **kwargs)) / file


def cpu_model():
    if platform.system() == "Linux":
        return re.search(r"^model name\s*:\s*(.*)$", Path("/proc/cpuinfo").read_text(), re.M)[1]
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    return platform.processor()


def machine(backend):
    # Without the last two, transformers runs Qwen3.5's linear attention in plain PyTorch.
    packages = {"transformers", "flash-linear-attention", "causal-conv1d"}
    facts = dict(cpu=cpu_model(), cpu_threads=os.cpu_count(), torch_threads=torch.get_num_threads(),
                 os=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                 cuda=torch.version.cuda, rocm=torch.version.hip,
                 packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                           if d.metadata["Name"] in packages})
    if hasattr(backend, "get_device_properties"):
        properties = backend.get_device_properties(0)
        facts.update(gpu=properties.name, vram_mib=properties.total_memory // 2**20)
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, default=Path(__file__).with_name("benchmark-cases.json"),
                        help="public cases (default: the Wikipedia set beside this script)")
    parser.add_argument("--dataset", help="registry evaluation suite instead of --case-file")
    parser.add_argument("--cases", type=int, help="first N cases by id")
    parser.add_argument("--model", default="0WD0/wime-syllable-40m",
                        help="model.safetensors, or a Hugging Face REPO[@REVISION]")
    parser.add_argument("--database", default="0WD0/wime-wanxiang",
                        help="wanxiang.db, or a Hugging Face dataset REPO[@REVISION]")
    parser.add_argument("--rescorer", default=QWEN, help="LM directory, or a Hugging Face REPO[@REVISION]")
    parser.add_argument("--rescorer-weight", type=float, default=1.5)
    parser.add_argument("--beam", type=int, action="append", help="beam widths (default: 32)")
    parser.add_argument("--device", default=accelerator())
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset:
        from dataset_registry import Registry
        path, dataset = Registry().evaluation(args.dataset)
        cases = json.loads(path.read_text())
    else:
        dataset = dict(path=args.case_file.name, sha256=fingerprint(args.case_file))
        cases = json.loads(args.case_file.read_text())["cases"]
    cases = sorted(cases, key=lambda c: c["id"])[:args.cases]
    specs = dict(model=args.model, database=args.database, rescorer=args.rescorer)
    model = fetch(args.model, "wime-syllable-40m", "model.safetensors")
    database = fetch(args.database, ".", "wanxiang.db", repo_type="dataset")
    lm = fetch(args.rescorer, "rescorers/" + args.rescorer.partition("@")[0].split("/")[-1])
    torch.set_num_threads(args.threads)
    backend = getattr(torch, args.device.split(":")[0])
    start = time.perf_counter()
    evaluator = Evaluator(model, device=args.device)
    rescorer = Rescorer(str(lm), args.rescorer_weight, device=args.device)
    units = Units.from_database(rescorer.tokenizer, database)
    backend.synchronize()
    report = dict(dataset=dataset, cases=len(cases), models=specs, model_sha256=fingerprint(model),
                  rescorer_weight=args.rescorer_weight, device=args.device, threads=args.threads,
                  machine=machine(backend), load_s=time.perf_counter() - start, semantics=__doc__, beams={})
    for beam in args.beam or [32]:
        decoder = Decoder(evaluator, rescorer, units, beam=beam)
        if hasattr(backend, "reset_peak_memory_stats"):
            backend.reset_peak_memory_stats()
        start = time.perf_counter()
        decoder.decode("评测预热", "nihk")
        backend.synchronize()
        warmup_ms = 1000 * (time.perf_counter() - start)
        times, correct = [], 0
        for case in cases:
            decoder.session = None
            for k in range(2, len(case["input"]) + 1):
                backend.synchronize()
                start = time.perf_counter()
                result = decode_composition(decoder, case["context"], case["input"][:k],
                                            window=evaluator.model.cfg.context)
                backend.synchronize()
                times.append(1000 * (time.perf_counter() - start))
            correct += result[0][0] in case["gold"]
        row = dict(warmup_ms=warmup_ms, keystrokes=len(times),
                   **{f"p{int(f * 100)}": percentile(times, f) for f in (.5, .9, .95, .99)},
                   max=max(times), accuracy=correct / len(cases))
        if hasattr(backend, "max_memory_reserved"):
            row.update(peak_allocated_mib=backend.max_memory_allocated() / 2**20,
                       peak_reserved_mib=backend.max_memory_reserved() / 2**20)
        report["beams"][beam] = row
        print(json.dumps({beam: row}), flush=True)
    # ru_maxrss is in KiB on Linux and bytes on macOS.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_mib"] = rss / (2**20 if platform.system() == "Darwin" else 2**10)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
