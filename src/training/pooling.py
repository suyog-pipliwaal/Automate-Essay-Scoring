import torch
import torch.nn as nn
import torch.nn.functional as F


class Pooling(nn.Module):
  def __init__(self,hidden_size:int, mode:str, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if mode == "attention":
      self.attn = nn.Linear(hidden_size, 1)
    self.mode = mode
  def forward(self, hidden_state, attention_mask):
    if self.mode == "cls":
      return hidden_state[:,0]
    mask = attention_mask.unsqueeze(-1)
    if self.mode == "mean":
      return hidden_state.mean(dim=1)
    if self.mode == "max":
      masked = hidden_state.masked_fill(mask == 0, -1e9)
      return masked.max(dim=1).values
    if self.mode == "attention":
      scores = self.attn(hidden_state).squeeze(-1)
      min_val = torch.finfo(scores.dtype).min
      scores = scores.masked_fill(attention_mask == 0, min_val)
      weights = torch.softmax(scores, dim=1).unsqueeze(-1)
      pooled = (hidden_state * weights).sum(dim=1)
      return pooled

    raise ValueError(f"unknow pooling mode:{self.mode}")