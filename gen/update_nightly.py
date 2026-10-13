#!/usr/bin/env python3
"""Refresh the nightly rime-wanxiang URLs and hashes in rime_wanxiang.bzl.

Run from anywhere; the script downloads the two nightly assets and rewrites
the ``_*_SHA256`` constants so a moving nightly build can be pinned again.
"""

import hashlib
import re
import urllib.request
from pathlib import Path

ASSETS = {
    "_DICT": "https://github.com/amzxyz/rime-wanxiang/releases/download/"
             "dict-nightly/rime-wanxiang-moqi-fuzhu.zip",
    "_GRAM": "https://github.com/amzxyz/RIME-LMDG/releases/download/"
             "LTS/wanxiang-lts-zh-hans.gram",
}


def download_sha256(url):
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main():
    source = Path(__file__).resolve().parent.parent / "rime_wanxiang.bzl"
    text = source.read_text(encoding="utf-8")
    for key, url in ASSETS.items():
        digest = download_sha256(url)
        text = re.sub(r'(%s_SHA256 = ")[^"]*(")' % key, r"\g<1>%s\g<2>" % digest,
                      text)
        print("%s %s %s" % (key, digest, url))
    source.write_text(text, encoding="utf-8")
    print("updated", source)


if __name__ == "__main__":
    main()
