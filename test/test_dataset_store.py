"""Exercise dataset identity against a real DVC cache and Parquet release."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from dataset_registry import Registry


class DatasetStore(unittest.TestCase):
    def test_registered_release_rejects_changed_data_and_pointer(self):
        with tempfile.TemporaryDirectory() as tmp, patch("dataset_registry.DATASET_PYTHON", sys.executable):
            root = Path(tmp)

            def dvc(*args):
                subprocess.run([sys.executable, "-m", "dvc", *args], cwd=root, check=True,
                               capture_output=True)

            dvc("init", "--no-scm")
            dvc("config", "cache.type", "copy")
            release = root / "release"
            release.mkdir()
            (release / "manifest.json").write_text('{}\n')
            for split in ("train", "dev"):
                (release / split).mkdir()
                pq.write_table(pa.Table.from_pylist([dict(gold=["你好"])]), release / split / "part.parquet")
            dvc("add", "release")
            catalog = root / "datasets.json"
            catalog.write_text('{"version":1,"datasets":{}}')
            registry = Registry(catalog, roots={"repo": root})
            registry.register(dict(id="release", kind="corpus", status="active", annotation="contextual",
                                   location=dict(root="repo", path="release"),
                                   storage=dict(type="dvc", metadata="release.dvc"),
                                   files=dict(manifest=dict(path="manifest.json"),
                                              **{s: dict(path=s, format="parquet") for s in ("train", "dev")})))
            directory, _, snapshot = registry.training("release")
            self.assertEqual(directory, release)
            self.assertEqual(snapshot["files"]["train"]["rows"], 1)
            self.assertNotIn("sha256", snapshot["files"]["train"])
            (release / "train/part.parquet").write_bytes(b"changed data")
            with self.assertRaisesRegex(ValueError, "checkout or repair"):
                registry.training("release")
            dvc("checkout", "--force")
            registry.training("release")
            pointer = root / "release.dvc"
            pointer.write_text(pointer.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed DVC dataset pointer"):
                registry.verify("release")


if __name__ == "__main__":
    unittest.main()
