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

### (任意) パッケージとして開発インストール
```
pip install -e .          # src/llm_from_scratch を import 可能にする
```

### Check environment libraries
```
python python_environment_check.py
```

## Directory layout

```
src/llm_from_scratch/     自前実装パッケージ
  tokenizer.py    (ch02)  正規表現ベースの簡易トークナイザ
  dataloader.py   (ch02)  スライディングウィンドウで (input, target) を作る DataLoader
  attention.py    (ch03)  Self / Causal / MultiHead attention
  model.py        (ch04)  GPTModel 本体 (TransformerBlock, LayerNorm, GELU, FeedForward)
  training.py     (ch05)  学習ループ・損失計算・温度/top-k サンプリング付き生成
scripts/
  train.py                学習 CLI エントリポイント
  generate.py             生成 CLI エントリポイント
data/book.txt             学習用テキスト
models/model.pth          学習済み重み (git 管理外)
pyproject.toml            src レイアウトのパッケージ定義
```

## Train a GPT on book.txt

```
python scripts/train.py                     # デフォルト（10 epoch）で学習し models/model.pth を保存
python scripts/train.py --num-epochs 20 --plot
python scripts/train.py --prompt "Every effort moves you"
```

学習中は epoch ごとにサンプル生成が表示され、終了後に重みが `models/model.pth` に保存される。

## Generate from a trained model

```
python scripts/generate.py --prompt "Every effort moves you" --max-new-tokens 80 --temperature 1.2 --top-k 40
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
