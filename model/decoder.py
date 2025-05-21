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
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embedding_dim):
        super(PositionalEncoding, self).__init__()

        self.embeddings = nn.Embedding(max_seq_len, embedding_dim)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.embeddings(positions)

        return x+pos_emb

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
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
    def __init__(self, embedding_dim, ff_embedding_dim, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_embedding_dim),
            nn.GELU(), # Gaussian Error Linear Units
            nn.Linear(ff_embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)

class ResidualBlock(nn.Module):
    def __init__(self, sub_layer, hidden_size):
        super(ResidualBlock, self).__init__()
        self.sub_layer = sub_layer
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, **kwargs):
        return self.norm(x+self.sub_layer(x, **kwargs) if kwargs else self.sub_layer(x))

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_embedding_dim, dropout=0.1):
        super().__init__()

        self.attn = ResidualBlock(
            MultiHeadSelfAttention(embedding_dim, num_heads), 
            embedding_dim
        )

        self.ffn = ResidualBlock(
            FeedForward(embedding_dim, ff_embedding_dim, dropout), 
            embedding_dim
        )

        # self.attn = MultiHeadSelfAttention(config.embedding_dim, config.num_heads, config.dropout)
        # self.ffn = FeedForward(config.embedding_dim, config.ff_embedding_dim, config.dropout)

        # self.ln1 = nn.LayerNorm(config.embedding_dim)
        # self.ln2 = nn.LayerNorm(config.embedding_dim)

    def forward(self, x, mask=None):

        x = self.attn(x)
        x = self.ffn(x)
        return x
        # Self-attention + residual
        # attn_out = self.attn(self.ln1(x), mask)
        # x = x + attn_out

        # # Feed-forward + residual
        # ffn_out = self.ffn(self.ln2(x))
        # x = x + ffn_out