"""The fused decoder: a conditional model or syllable transducer plus a weighted HF language model."""


def add_fusion_arguments(parser):
    parser.add_argument("--rescorer", help="local HF language model weights")
    parser.add_argument("--rescorer-weight", type=float, default=1.5,
                        help="language-model weight in verify and decode fusion")


def make_rescorer(args):
    from rescorer import Rescorer
    return Rescorer(args.rescorer, args.rescorer_weight) if args.rescorer else None


def make_evaluator(path, device="cuda"):
    """The evaluator a checkpoint's metadata names: a transducer or a conditional model."""
    from safetensors import safe_open
    import transducer
    with safe_open(str(path), "pt") as f:
        if f.metadata().get("format") == transducer.FORMAT:
            return transducer.Evaluator(path, device)
    from inference import ConditionalEvaluator
    return ConditionalEvaluator(path, device=device)


def make_decoder(args, evaluator, rescorer):
    if rescorer is None or not args.database:
        return None
    from decoder import Decoder, Units
    from decoding_model import readings
    import transducer
    kind = transducer.decoder_class(Decoder) if isinstance(evaluator, transducer.Evaluator) else Decoder
    return kind(evaluator, rescorer, Units(rescorer.tokenizer, readings(args.database)))
