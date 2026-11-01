"""Relocatable worker sources and pinned pinyin data; PyTorch uses host CUDA."""

def _gpu_runtime_impl(ctx):
    output = ctx.actions.declare_directory("wime-cuda")
    prefix = ctx.label.package + "/"
    files = {source.short_path[len(prefix):]: source.path for source in ctx.files.srcs}
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
    args = ctx.actions.args()
    args.add(output.path)
    for relative, source in files.items():
        args.add(relative)
        args.add(source)
    ctx.actions.run_shell(
        command = """set -eu
destination="$1"
shift
while [ "$#" -gt 0 ]; do
    output="$destination/$1"
    mkdir -p "$(dirname "$output")"
    cp "$2" "$output"
    case "$1" in gpu-worker.py) chmod 755 "$output";; *) chmod 644 "$output";; esac
    shift 2
done
""",
        arguments = [args],
        inputs = ctx.files.srcs + ctx.files.pypinyin,
        outputs = [output],
        mnemonic = "WimeCudaRuntime",
    )
    return [DefaultInfo(files = depset([output]))]

gpu_runtime = rule(
    implementation = _gpu_runtime_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "pypinyin": attr.label(default = "@wime_pypinyin//:files"),
    },
)
