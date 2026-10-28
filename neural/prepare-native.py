#!/usr/bin/env python3
"""Build the imported llama.cpp and install MiniCPM assets outside Wdoom snapshots.

Run with Wime's Python. --model-dir is a local MiniCPM5-2B HF download.
An existing BF16 --gguf can be reused; otherwise conversion uses --convert-python
(which needs the imported converter's requirements). No service is started.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

NEURAL = Path(__file__).resolve().parent
LLAMA = NEURAL.parents[2] / "third_party/llama.cpp"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_file(source, target):
    """Copy atomically; an existing different model must never be overwritten."""
    sha = digest(source)
    if target.exists():
        if digest(target) != sha:
            raise ValueError(f"Different installed asset already exists: {target}")
        return sha
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        if digest(temporary) != sha:
            raise ValueError("Asset changed while copying")
        os.chmod(temporary, 0o444)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha


def validate_gguf(path, config):
    # Use the converter shipped alongside the imported native runtime.
    sys.path.insert(0, str(LLAMA / "gguf-py"))
    from gguf import GGUFReader, GGMLQuantizationType, LlamaFileType
    reader = GGUFReader(path)
    expected = {"general.architecture": "llama", "general.file_type": int(LlamaFileType.MOSTLY_BF16),
                "llama.block_count": config["num_hidden_layers"],
                "llama.embedding_length": config["hidden_size"],
                "llama.attention.head_count": config["num_attention_heads"],
                "llama.attention.head_count_kv": config["num_key_value_heads"]}
    for key, value in expected.items():
        if key not in reader.fields or reader.fields[key].contents() != value:
            raise ValueError(f"GGUF does not match the BF16 MiniCPM configuration: {key}")
    if len(reader.fields["tokenizer.ggml.tokens"].data) != config["vocab_size"]:
        raise ValueError("GGUF tokenizer size mismatch")
    if any(t.tensor_type not in (GGMLQuantizationType.BF16, GGMLQuantizationType.F32)
           for t in reader.tensors):
        raise ValueError("Only the evaluated BF16 model is supported")


def build(args):
    settings = dict(cuda_architectures=args.cuda_architectures,
                    nvcc=subprocess.check_output([args.nvcc, "--version"], text=True),
                    cxx=subprocess.check_output([os.environ.get("CXX", "c++"), "--version"], text=True))
    inputs = [NEURAL / "native/CMakeLists.txt", NEURAL / "native/llama_bridge.cpp", LLAMA / "CMakeLists.txt"]
    for name in ("cmake", "src", "include", "ggml", "vendor"):
        inputs.extend(p for p in (LLAMA / name).rglob("*") if p.is_file())
    sources = {str(p.relative_to(NEURAL.parents[2])): digest(p) for p in sorted(inputs)}
    identity = dict(settings=settings, sources=sources)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    prefix = args.root / "native" / key
    if prefix.exists():
        if not (prefix / "lib/libwime_llama.so").is_file():
            raise ValueError(f"Incomplete native installation: {prefix}")
        return prefix
    build_dir = args.build_root / key
    subprocess.run(["cmake", "-S", str(NEURAL / "native"), "-B", str(build_dir), "-G", "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON",
                    "-DGGML_NATIVE=OFF", "-DCMAKE_CUDA_COMPILER=" + args.nvcc,
                    "-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_architectures], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--parallel", str(args.jobs)], check=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=prefix.parent) as temporary:
        stage = Path(temporary) / "runtime"
        subprocess.run(["cmake", "--install", str(build_dir), "--prefix", str(stage)], check=True)
        # The installed tree must load without build-directory RPATHs.
        subprocess.run([sys.executable, "-c",
                        "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); assert lib.wime_llama_abi_version()==1",
                        str(stage / "lib/libwime_llama.so")], check=True)
        (stage / "manifest.json").write_text(json.dumps(dict(key=key, **identity), indent=2) + "\n")
        stage.rename(prefix)
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("XDG_DATA_HOME", Path.home()/".local/share"))/"wime")
    parser.add_argument("--build-root", type=Path, default=Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"wime-native")
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "/opt/cuda/bin/nvcc")
    parser.add_argument("--cuda-architectures", required=True, help="explicit CUDA SM list, e.g. 89 for RTX 4090")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--convert-python", default=sys.executable)
    args = parser.parse_args()
    args.root, args.model_dir = args.root.expanduser().resolve(), args.model_dir.expanduser().resolve()
    config = json.loads((args.model_dir / "config.json").read_text())
    if (config.get("model_type"), config.get("num_hidden_layers"), config.get("vocab_size")) != ("llama", 42, 130560):
        parser.error("--model-dir must contain MiniCPM5-2B")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "native-install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination = args.root / "rescorers/MiniCPM5-2B"
        with tempfile.TemporaryDirectory(prefix="wime-convert-", dir=args.root) as temporary:
            gguf = args.gguf or destination / "model.bf16.gguf"
            if not gguf.exists():
                if args.gguf:
                    raise FileNotFoundError(gguf)
                gguf = Path(temporary) / "model.gguf"
                subprocess.run([args.convert_python, str(LLAMA / "convert_hf_to_gguf.py"),
                                str(args.model_dir), "--outfile", str(gguf), "--outtype", "bf16"], check=True)
            validate_gguf(gguf, config)
            prefix = build(args)
            files = {}
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                         "generation_config.json", "chat_template.jinja", "LICENSE"):
                source = args.model_dir / name
                if source.exists():
                    files[name] = install_file(source, destination / name)
                elif name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                    raise FileNotFoundError(source)
            files["model.bf16.gguf"] = install_file(gguf, destination / "model.bf16.gguf")
            manifest = Path(temporary) / "installed.json"
            manifest.write_text(json.dumps(dict(format="BF16", files=files), indent=2) + "\n")
            install_file(manifest, destination / "installed.json")
        # A failed build or model copy cannot replace the last good library.
        with tempfile.TemporaryDirectory(prefix=".select-", dir=prefix.parent) as temporary:
            link = Path(temporary) / "current"
            link.symlink_to(prefix)
            os.replace(link, prefix.parent / "current")
    print(json.dumps(dict(library=str(prefix / "lib/libwime_llama.so"), model=str(destination / "model.bf16.gguf"))))


if __name__ == "__main__":
    main()
