import torch.nn as nn

class Normalization(nn.Module):
  
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-06)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    