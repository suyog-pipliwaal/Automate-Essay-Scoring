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
      score = self.attn(hidden_state).squeeze(-1)
      score = score.masked_fill(attention_mask==0, -1e9)
      weights = F.softmax(score, dim=1).unsqueeze(-1)
      return (hidden_state*weights).sum(dim=1)
    raise ValueError(f"unknow pooling mode:{self.mode}")