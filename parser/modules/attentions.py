from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout: float = 0, log_softmax=False) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.LogSoftmax(dim=-1) if log_softmax else nn.Softmax(dim=-1)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        # Calculate scale
        sqrt_dim = math.sqrt(q.shape[-1])

        # Dot product
        if q.dim() == 3:
            attn = torch.bmm(q, k.transpose(1, 2))
        elif q.dim() == 2:
            attn = torch.mm(q, k.transpose(0, 1))
        else:
            raise ValueError("q must be 2 or 3 dimensional")
        # Scaled dot product
        attn = attn / sqrt_dim

        # Masking
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        # Softmax & Dropout
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        if v is None:
            return None, attn

        # Weighted sum for value vector
        if q.dim() == 3:
            output = torch.bmm(attn, v)
        elif q.dim() == 2:
            output = torch.mm(attn, v)
        return output, attn
