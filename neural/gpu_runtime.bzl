"""Relocatable worker sources and pinned pinyin data; PyTorch uses host CUDA."""

def _gpu_runtime_impl(ctx):
    output = ctx.actions.declare_directory("wime-cuda")
    manifest = ctx.actions.declare_file(ctx.label.name + "-files.json")
    files = {source.basename: source.path for source in ctx.files.srcs}
    for source in ctx.files.pypinyin:
        # Wheel members retain their package/data/metadata paths.
        parts = source.path.split("/")
        matched = False
        for i, part in enumerate(parts):
            if part == "pypinyin" or part.endswith(".dist-info"):
                files["/".join(parts[i:])] = source.path
                matched = True
                break
        if not matched:
            fail("Unexpected pypinyin wheel member: " + source.path)
    ctx.actions.write(manifest, json.encode(files))
    ctx.actions.run(
        executable = ctx.executable._bundle,
        arguments = [manifest.path, output.path],
        inputs = [manifest] + ctx.files.srcs + ctx.files.pypinyin,
        outputs = [output],
        mnemonic = "WimeCudaRuntime",
    )
    return [DefaultInfo(files = depset([output]))]

gpu_runtime = rule(
    implementation = _gpu_runtime_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "pypinyin": attr.label(default = "@wime_pypinyin//:files"),
        "_bundle": attr.label(default = "//emacs-packages/wime/neural:bundle-gpu", executable = True, cfg = "exec"),
    },
)
