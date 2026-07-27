# LLM-from-scratch

「[Build a Large Language Model From Scratch](https://github.com/rasbt/LLMs-from-scratch)」を
章ごとに実装。tokenizer から GPT 本体・学習ループまでを `src/llm_from_scratch/` にパッケージ化している。

## Environment set up (conda)

### Create environment and install libraries
```
conda create -n llms python=3.10
conda activate llms
pip install -r requirements.txt
```

### Install as llm-from-scratch as a package
```
pip install -e .
```

### Check environment libraries
```
python python_environment_check.py
```

## Directory layout

```
src/llm_from_scratch/     Package
  tokenizer.py    (ch02)  Simple tokenizer with regexp
  dataloader.py   (ch02)  Create (input, target) DataLoader using sliding window
  attention.py    (ch03)  Self / Causal / MultiHead attention
  model.py        (ch04)  GPTModel (TransformerBlock, LayerNorm, GELU, FeedForward)
  training.py     (ch05)  trainig loop・loss calc・temperature/top-k
scripts/
  train.py
  generate.py
data/book.txt             学習用テキスト
pyproject.toml            src レイアウトのパッケージ定義
```

## Train a GPT on book.txt

```
python scripts/train.py                     # デフォルト（10 epoch）で学習し models/model.pth を保存
python scripts/train.py --num-epochs 20 --plot
python scripts/train.py --prompt "Every effort moves you"
```

学習中は epoch ごとにサンプル生成が表示され、終了後に重みが `models/model.pth` に保存される。

## Train a GPT on Project Gutenberg Dataset
1. Clone `gutenberg` dataset
```
git clone https://github.com/pgcorpus/gutenberg.git
```

2. Navigate to reposigory
```
cd gutenberg
```

3. Install required packages
```
pip install -r requirements.txt
```

4. Download data
```
python get_data.py
cd ..
```

5. Prepare dataset
```
python scripts/prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

6. Run train script
```
python scripts/train.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

## Train envs

- **GPU**: RTX 3090 24GB
- **Dataset**: Project Gutenberg — 57GB, 50,896 files

### Hyperparameters

| 項目 | 値 |
|---|---|
| batch size | 4 |
| epochs | 1 |
| optimizer | AdamW (`lr=5e-4`, `weight_decay=0.1`) |
| context length / stride | 1024 / 1024（非オーバーラップ） |
| train / val split | 0.90 / 0.10（book 単位で毎回作り直し） |
| eval freq | 100 step |
| seed | 123 |

### Model config

`GPT_CONFIG_124M`（[src/llm_from_scratch/model.py](src/llm_from_scratch/model.py)）

| key | 値 | 意味 |
|---|---|---|
| `vocab_size` | 50257 | 語彙数（GPT-2 BPE） |
| `context_length` | 1024 | コンテキスト長 |
| `emb_dim` | 768 | 埋め込み次元 |
| `n_heads` | 12 | attention ヘッド数 |
| `n_layers` | 12 | Transformer ブロック数 |
| `drop_rate` | 0.1 | ドロップアウト率 |
| `qkv_bias` | False | Q/K/V 線形層のバイアス |

## Train process
This is the train process until 200k step.
<img width="690" height="335" alt="スクリーンショット 2026-07-27 23 23 43" src="https://github.com/user-attachments/assets/416a74b4-9f37-48b1-b533-8f35991e4c85" />


## Generate from a trained model

```
python generate.py --prompt "Every effort moves you" --max-new-tokens 80 --temperature 1.2 --top-k 40
```

## ライブラリとして使う

```python
import llm_from_scratch as L

model = L.GPTModel(L.GPT_CONFIG_124M)
loader = L.create_dataloader(text)
L.train_model_simple(model, train_loader, val_loader, optimizer, device,
                     num_epochs=10, eval_freq=5, eval_iter=5,
                     start_context="Every effort", tokenizer=tokenizer)
```
