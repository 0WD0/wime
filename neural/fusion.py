"""The fused decoder: the conditional model plus a weighted HF language model."""


def add_fusion_arguments(parser):
    parser.add_argument("--rescorer", help="local HF language model weights")
    parser.add_argument("--rescorer-weight", type=float, default=1.5,
                        help="language-model weight in verify and decode fusion")


def make_rescorer(args):
    from rescorer import Rescorer
    return Rescorer(args.rescorer, args.rescorer_weight) if args.rescorer else None


def make_decoder(args, evaluator, rescorer):
    if rescorer is None or not args.database:
        return None
    from decoder import Decoder, Units
    from decoding_model import readings
    return Decoder(evaluator, rescorer, Units(rescorer.tokenizer, readings(args.database)))
