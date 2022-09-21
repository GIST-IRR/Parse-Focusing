from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout: float = 0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        sqrt_dim = math.sqrt(q.shape[-1])
        if q.dim() == 3:
            attn = torch.bmm(q, k.transpose(1, 2))
        elif q.dim() == 2:
            attn = torch.mm(q, k.transpose(0, 1))
        else:
            raise ValueError("q must be 2 or 3 dimensional")
        attn = attn / sqrt_dim

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        if q.dim() == 3:
            output = torch.bmm(attn, v)
        elif q.dim() == 2:
            output = torch.mm(attn, v)
        return output, attn
