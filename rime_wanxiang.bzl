"""Fetch the nightly rime-wanxiang dictionaries and grammar at build time.

The Moqi auxiliary codes only exist in the release artifacts, not in the
git tree, so the release archive is downloaded and extracted.  The grammar
model is a separate nightly-updated asset from amzxyz/RIME-LMDG.

Run `gen/update_nightly.py` to refresh the URLs and hashes.
"""

_DICT_URL = "https://github.com/amzxyz/rime-wanxiang/releases/download/dict-nightly/rime-wanxiang-moqi-fuzhu.zip"
_DICT_SHA256 = "07ed550c6cb48ee2adb872fd4680b84535f620bc3311dd74e33ac80a6263f618"

_GRAM_URL = "https://github.com/amzxyz/RIME-LMDG/releases/download/LTS/wanxiang-lts-zh-hans.gram"
_GRAM_SHA256 = "d26da3d6fe567bcf4aad8790475356997d8bc0886b087b292c5063c0adc12bd0"


def _rime_wanxiang_impl(ctx):
    ctx.download_and_extract(url = _DICT_URL, sha256 = _DICT_SHA256)
    ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])

exports_files([
    "wanxiang_algebra.yaml",
    "wanxiang_pro.dict.yaml",
])

filegroup(
    name = "sources",
    srcs = glob(["dicts/**"]),
)
""")


def _rime_wanxiang_gram_impl(ctx):
    ctx.download(url = _GRAM_URL, output = "wanxiang-lts-zh-hans.gram",
                 sha256 = _GRAM_SHA256)
    ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])

exports_files(["wanxiang-lts-zh-hans.gram"])
""")


_rime_wanxiang = repository_rule(implementation = _rime_wanxiang_impl)
_rime_wanxiang_gram = repository_rule(implementation = _rime_wanxiang_gram_impl)


def _rime_wanxiang_extension_impl(ctx):
    for mod in ctx.modules:
        for tag in mod.tags.repository:
            _rime_wanxiang(name = tag.name)
        for tag in mod.tags.gram:
            _rime_wanxiang_gram(name = tag.name)


rime_wanxiang = module_extension(
    implementation = _rime_wanxiang_extension_impl,
    tag_classes = {
        "repository": tag_class(attrs = {"name": attr.string(mandatory = True)}),
        "gram": tag_class(attrs = {"name": attr.string(mandatory = True)}),
    },
)
