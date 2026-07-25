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
