"""Fetch the pinned rime-wanxiang Moqi release at build time.

The Moqi auxiliary codes only exist in the release artifacts, not in the
git tree, so the release archive is downloaded and extracted.  Only the
dictionaries and the algebra file are used.
"""

_URL = "https://github.com/amzxyz/rime-wanxiang/releases/download/v18.1.2/rime-wanxiang-moqi-fuzhu.zip"
_SHA256 = "ec217ee9e4abcabe752dae7164f79960cf310385ff1569585c91b73bfa3823bf"


def _rime_wanxiang_impl(ctx):
    ctx.download_and_extract(url = _URL, sha256 = _SHA256)
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


_rime_wanxiang = repository_rule(implementation = _rime_wanxiang_impl)


def _rime_wanxiang_extension_impl(ctx):
    for mod in ctx.modules:
        for tag in mod.tags.repository:
            _rime_wanxiang(name = tag.name)


rime_wanxiang = module_extension(
    implementation = _rime_wanxiang_extension_impl,
    tag_classes = {
        "repository": tag_class(attrs = {"name": attr.string(mandatory = True)}),
    },
)
