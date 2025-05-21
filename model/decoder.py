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


class OutputLayer(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.proj(x)  # (B, T, V)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim))

        # initialize positional_embedding tensor using normal distribution
        nn.init.normal_(self.position_embedding, std=0.02)

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        token_emb = self.token_embedding(input_ids)  # (B, T, D)
        seq_len = input_ids.size(1)
        pos_emb = self.position_embedding[:, :seq_len, :]  # (1, T, D)
        return self.dropout(token_emb + pos_emb)

def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
    return mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ✅ proper check
mask = create_causal_mask(10).to(device)   


# model = GPT2Decoder(
#     vocab_size=1000,
#     embedding_dim=64,
#     max_seq_len=50,
#     num_heads=4,
#     ff_dim=256,
#     num_layers=2
# )

# input_ids = torch.randint(0, 1000, (8, 10))
# labels = input_ids.clone()

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss()

# for _ in range(350):
#     logits = model(input_ids)
#     loss = criterion(logits.view(-1, 1000), labels.view(-1))
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     print(f"Loss: {loss.item():.4f}")


