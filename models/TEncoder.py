import copy
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(5000, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TEncoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, c1, c2, n_heads=4, d_ff=2048, dropout=0.2, N=2):
        super(TEncoder, self).__init__()
        self.layers = clones(TEncoderLayer(c2, n_heads, d_ff, dropout), N)  # layer 表示一整个encoder
        self.norm = nn.LayerNorm(c2)
        self.position = PositionalEncoding(c2, dropout)
        # self.position = nn.Linear(c1,c2)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        b, _, w, h = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        position = self.position(x)

        x = x + position
        for layer in self.layers:
            x = layer(x, mask=None)
        return self.norm(x).permute(1, 2, 0).reshape(b, -1, w, h)


class TEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # Position-wise feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head self-attention
        residual = x
        x = self.norm2(x)
        x, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.dropout(x)
        x += residual

        # Position-wise feedforward network
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x += residual
        return x
