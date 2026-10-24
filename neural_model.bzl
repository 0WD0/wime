"""Pinned model artifact; development sources live in third_party/chinese-ime-lm."""

def _model_impl(ctx):
    ctx.download(
        url = "https://github.com/metasequoiaime/chinese-ime-lm/releases/download/model-v1/sentence-model.safetensors",
        output = "sentence-model.safetensors",
        sha256 = "86ac529510cb3b4968a5e6ade83ec8080f5b34a0a681e75c362accbbd387d1c1",
    )
    ctx.file("BUILD.bazel", 'exports_files(["sentence-model.safetensors"], visibility = ["//visibility:public"])\n')

_model = repository_rule(implementation = _model_impl)

def _extension_impl(_ctx):
    _model(name = "wime_neural_model")

wime_neural_model = module_extension(implementation = _extension_impl)
