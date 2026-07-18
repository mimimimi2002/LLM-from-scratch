import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDataset(Dataset):
  def __init__(self, txt, tokenizer, max_length, stride = 1):
    self.input_ids = []
    self.target_ids = []

    token_ids = tokenizer.encode(txt)

    for i in range(0, len(txt) - max_length, stride):
      self.input_ids.append(token_ids[i: i + max_length])
      self.target_ids.append(token_ids[i + stride: i + max_length + stride])

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
      return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size=4, max_length=256,
  stride=128, shuffle=True, drop_last=True, num_workers=0):

  # vocab = SimpleTokenizer.create_vocabulary("./book.txt")
  # tokenizer = SimpleTokenizer(vocab)
  tokenizer = tiktoken.get_encoding("gpt2")
  dataset = GPTDataset(txt, tokenizer, max_length, stride)
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=0
  )
  return dataloader

if __name__ == "__main__":
  with open("book.txt", "r") as f:
    raw_text = f.read()
  dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
  data_iter = iter(dataloader)
  first_batch = next(data_iter)
  print(first_batch)