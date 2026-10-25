"""Materialize an immutable CUDA worker without links into the source checkout."""
import json
from pathlib import Path
import shutil
import sys


def main():
    manifest, destination = map(Path, sys.argv[1:])
    for relative, source in json.loads(manifest.read_text()).items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("bundle path escapes destination")
        output = destination / path
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        output.chmod(0o755 if relative == "gpu-worker.py" else 0o644)


if __name__ == "__main__":
    main()
