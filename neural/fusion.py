"""The fused decoder: the syllable transducer plus a weighted HF language model."""


def accelerator():
    """The first GPU backend this PyTorch build can use: NVIDIA and AMD (ROCm)
    GPUs are "cuda", Intel "xpu", Apple "mps"; otherwise the CPU."""
    import torch
    return next((name for name, backend in (("cuda", torch.cuda), ("xpu", torch.xpu), ("mps", torch.backends.mps))
                 if backend.is_available()), "cpu")


def add_fusion_arguments(parser):
    parser.add_argument("--device", default=accelerator(), help="PyTorch device (default: the first GPU found)")
    parser.add_argument("--rescorer", help="local HF language model weights")
    parser.add_argument("--rescorer-weight", type=float, default=1.5,
                        help="language-model weight in fused decoding")


def make_rescorer(args):
    from rescorer import Rescorer
    return Rescorer(args.rescorer, args.rescorer_weight, args.device) if args.rescorer else None


def make_decoder(args, evaluator, rescorer):
    if rescorer is None or not args.database:
        return None
    from decoder import Decoder, Units
    from syllables import database_layout
    evaluator.syllable_layout = database_layout(args.database)
    return Decoder(evaluator, rescorer, Units.from_database(rescorer.tokenizer, args.database))
