import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from llm_from_scratch import (
    GPTModel, GPT_CONFIG_124M,
    download_and_load_gpt2, load_weights_into_gpt,
)


def main():
    parser = argparse.ArgumentParser(description="Download GPT-2 weights into a GPTModel")
    parser.add_argument("--model-size", default="124M", choices=["124M", "355M", "774M", "1558M"])
    parser.add_argument("--models-dir", default=os.path.join(REPO_ROOT, "models", "gpt2"))
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "models", "gpt2-124M.pth"))
    args = parser.parse_args()

    settings, params = download_and_load_gpt2(args.model_size, args.models_dir)

    model = GPTModel(GPT_CONFIG_124M)
    load_weights_into_gpt(model, params)
    torch.save(model.state_dict(), args.out)
    print(f"Saved GPT-2 ({args.model_size}) weights to {args.out}")


if __name__ == "__main__":
    main()