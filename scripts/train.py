"""book.txt 上で小さな GPT を自前学習し、生成デモ + 重み保存まで行う CLI。

使い方:
    python scripts/train.py                    # デフォルト設定で学習
    python scripts/train.py --num-epochs 20
    python scripts/train.py --prompt "Every effort"
"""

import argparse
import os
import sys

import tiktoken
import torch

# src/ を検索パスに通す (エントリスクリプトのみで行う bootstrap)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from llm_from_scratch import (
    GPTModel,
    create_dataloader,
    train_model_simple,
    generate,
    text_to_token_ids,
    token_ids_to_text,
    plot_losses,
)

# GPT-2 (124M) と同じ構成。context_length だけ学習デモ用に短くしてある。
GPT_CONFIG = {
    "vocab_size": 50257,    # tiktoken gpt2 の語彙数
    "context_length": 256,  # 学習時の系列長 (本家 1024 より短くして軽量化)
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

DEFAULT_BOOK = os.path.join(REPO_ROOT, "data", "book.txt")
DEFAULT_OUT = os.path.join(REPO_ROOT, "models", "model.pth")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Train a small GPT on book.txt (ch05)")
    parser.add_argument("--book", default=DEFAULT_BOOK)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--prompt", default="Every effort moves you")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(123)
    device = get_device()
    print(f"Using device: {device}")

    # --- データ読み込み & train/val 分割 ---
    with open(args.book, "r", encoding="utf-8") as f:
        text_data = f.read()

    split_idx = int(args.train_ratio * len(text_data))
    train_data, val_data = text_data[:split_idx], text_data[split_idx:]

    ctx = GPT_CONFIG["context_length"]
    train_loader = create_dataloader(
        train_data, batch_size=args.batch_size, max_length=ctx,
        stride=ctx, drop_last=True, shuffle=True, num_workers=0)
    val_loader = create_dataloader(
        val_data, batch_size=args.batch_size, max_length=ctx,
        stride=ctx, drop_last=False, shuffle=False, num_workers=0)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- モデル & オプティマイザ ---
    model = GPTModel(GPT_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tokenizer = tiktoken.get_encoding("gpt2")

    # --- 学習 ---
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=5, eval_iter=5,
        start_context=args.prompt, tokenizer=tokenizer)

    # --- 生成デモ (温度 + top-k) ---
    model.eval()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(args.prompt, tokenizer).to(device),
        max_new_tokens=50, context_size=ctx, top_k=25, temperature=1.0)
    print("\n=== Generated ===")
    print(token_ids_to_text(token_ids, tokenizer))

    # --- 保存 ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"\nSaved weights to {args.out}")

    if args.plot:
        import numpy as np
        epochs_tensor = torch.linspace(0, args.num_epochs, len(train_losses))
        out_plot = os.path.join(REPO_ROOT, "models", "loss-plot.pdf")
        plot_losses(epochs_tensor, np.array(tokens_seen), train_losses, val_losses, out_plot)
        print(f"Saved loss curve to {out_plot}")


if __name__ == "__main__":
    main()
