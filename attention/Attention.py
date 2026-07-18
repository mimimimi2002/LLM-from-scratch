import torch.nn as nn
import torch
class SelfAttention(nn.Module):
  def __init__(self, d_in, d_out, qkv_bias=False):
    super().__init__()
    self.d_out = d_out
    self.W_query = nn.Linear(d_in, d_out, qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, qkv_bias)

  def forward(self, x):
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    attn_scores = queries @ keys.T

    # normalization
    atten_weights = torch.softmax(
      attn_scores / keys.shape[-1]**0.5, dim=-1
    )

    # multiply value
    context_vec = atten_weights @ values
    return context_vec

if __name__ == "__main__":
  inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
  )
  torch.manual_seed(123)
  d_in = inputs.shape[1]
  d_out = 2
  self_attention = SelfAttention(d_in, d_out)
  print(self_attention(inputs))