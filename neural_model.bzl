"""Pinned local conditional checkpoint and the CUDA worker's Python data."""

# Selected on development data and checked by test/ranking-measurements.json.
# Small suite regressions are recorded; never follow a mutable training output.
_SHA256 = "1c9b9e780ea833639eab03ed331e5876ed0afd0b07514634f0f32306a4f4059e"

def _model_impl(ctx):
    data = ctx.getenv("XDG_DATA_HOME") or (ctx.getenv("HOME") + "/.local/share")
    source = ctx.getenv("WIME_CONDITIONAL_MODEL") or (data + "/wime/models/" + _SHA256 + ".safetensors")
    path = ctx.path(source)
    if not path.exists:
        fail("Install the evaluated Wime checkpoint at %s, or set WIME_CONDITIONAL_MODEL to its path. Required SHA256: %s" % (source, _SHA256))
    ctx.watch(path)
    ctx.download(
        url = "file://" + str(path),
        output = "conditional-model.safetensors",
        sha256 = _SHA256,
    )
    ctx.file("BUILD.bazel", 'exports_files(["conditional-model.safetensors"], visibility = ["//visibility:public"])\n')

_model = repository_rule(implementation = _model_impl, local = True, configure = True)

def _pypinyin_impl(ctx):
    ctx.download_and_extract(
        url = "https://files.pythonhosted.org/packages/b9/7b/4cabc76fcc21c3c7d5c671d8783984d30ac9d3bb387c4ba784fca3cdfa3a/pypinyin-0.55.0-py2.py3-none-any.whl",
        sha256 = "d53b1e8ad2cdb815fb2cb604ed3123372f5a28c6f447571244aca36fc62a286f",
        type = "zip",
    )
    ctx.file("BUILD.bazel", """
filegroup(
    name = "files",
    srcs = glob(["pypinyin/**", "pypinyin-0.55.0.dist-info/**"]),
    visibility = ["//visibility:public"],
)
""")

_pypinyin = repository_rule(implementation = _pypinyin_impl)

def _extension_impl(_ctx):
    _model(name = "wime_neural_model")
    _pypinyin(name = "wime_pypinyin")

wime_neural_model = module_extension(implementation = _extension_impl)
