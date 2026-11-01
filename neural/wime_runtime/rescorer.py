"""General causal language model shared by Chinese and Japanese decoding."""
import torch


class Rescorer:
    def __init__(self, path, weight, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # CPUs without native BF16 matmul run FP32 several times faster.
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        self.weight, self.device = weight, device
        self.pad = self.tokenizer.pad_token_id or 0
