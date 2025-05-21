# model/gpt2.py (add this later)

import torch
import torch.nn as nn
from decoder import DecoderLayer
from embeddings import GPT2Embedding
from decoder import EmbeddingLayer, OutputLayer

class GPT2Model(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            ff_embedding_dim,
            max_seq_len,
            num_heads,
            num_layers,
            dropout = 0.1,
        ):
        super().__init__()
        self.embeddings = GPT2Embedding(vocab_size, embedding_dim, max_seq_len)
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, ff_embedding_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.embeddings = EmbeddingLayer(vocab_size, embedding_dim, max_seq_len)
        self.lm_head = OutputLayer(embedding_dim, vocab_size)

    def forward(self, x, mask=None):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)
