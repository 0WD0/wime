"""Chinese text decoding and the independently selectable legacy fusion."""


def add_backend_arguments(parser):
    parser.add_argument("--backend", choices=("qwen-hf", "minicpm-native", "legacy"), default="qwen-hf")
    parser.add_argument("--rescorer", help="local HF weights for Qwen, or matching HF tokenizer for MiniCPM")
    parser.add_argument("--rescorer-weight", type=float, default=1.5,
                        help="LM weight for verify and legacy fusion; text decoding ranks by LM probability")
    parser.add_argument("--native-library", help="Wime llama bridge shared library")
    parser.add_argument("--native-model", help="BF16 MiniCPM GGUF matching --rescorer")


def validate_backend(args, parser):
    if args.backend == "legacy" and not (args.rescorer and args.database):
        parser.error("legacy requires --rescorer and --database")
    if args.backend == "minicpm-native":
        if not (args.native_library and args.native_model and args.rescorer and args.database):
            parser.error("minicpm-native requires --native-library, --native-model, --rescorer and --database")
    elif args.native_library or args.native_model:
        parser.error("native assets require --backend minicpm-native")


def make_rescorer(args):
    if args.backend == "minicpm-native":
        from native_rescorer import NativeRescorer
        return NativeRescorer(args.rescorer, args.rescorer_weight,
                              args.native_library, args.native_model)
    from rescorer import Rescorer
    return Rescorer(args.rescorer, args.rescorer_weight) if args.rescorer else None


def make_decoder(args, evaluator, rescorer):
    if rescorer is None or not args.database:
        return None
    from decoding_model import readings
    if args.backend == "legacy":
        from decoder import Decoder, Units
        return Decoder(evaluator, rescorer, Units(rescorer.tokenizer, readings(args.database)))
    from text_decoder import TextDecoder
    backend = dict(lm_backend=rescorer.backend) if args.backend == "minicpm-native" else {}
    return TextDecoder(evaluator, rescorer, readings(args.database), **backend)
