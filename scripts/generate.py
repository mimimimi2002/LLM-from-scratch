import argparse
import os
import sys

import tiktoken
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from llm_from_scratch import (
    GPTModel,
    generate,
    text_to_token_ids,
    token_ids_to_text,
)

# scripts/train.py と同じ学習構成
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

DEFAULT_WEIGHTS = os.path.join(REPO_ROOT, "models", "model.pth")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--prompt", default="Every effort moves you")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = get_device()
    model = GPTModel(GPT_CONFIG).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(args.prompt, tokenizer).to(device),
        max_new_tokens=args.max_new_tokens,
        context_size=GPT_CONFIG["context_length"],
        top_k=args.top_k,
        temperature=args.temperature,
    )
    print(token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
