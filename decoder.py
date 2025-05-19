# model/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
B: Batch size
T: sequence length
D: size of embedding
H: number of attention heads
d: head dimension
"""

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.size()

        qkv = self.qkv_proj(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # computing attention
        # step 1: q * k matrix multiplication
        # step 2: scaling
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # step 3: masking future tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # step 4: apply softmax function to normalize the compatibility matrix
        # giving us the attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # step 5: computing (output / context) matrix
        out = attn @ v  # (B, H, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, hidden_size, ff_hidden_size, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_hidden_size),
            nn.GELU(), # Gaussian Error Linear Units
            nn.Linear(ff_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadSelfAttention(config.hidden_size, config.num_heads, config.dropout)
        self.ffn = FeedForward(config.hidden_size, config.ff_hidden_size, config.dropout)

        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, mask=None):
        # Self-attention + residual
        attn_out = self.attn(self.ln1(x), mask)
        x = x + attn_out

        # Feed-forward + residual
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out

        return x
