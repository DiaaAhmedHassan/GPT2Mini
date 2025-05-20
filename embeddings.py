# model/embeddings.py

import torch
import torch.nn as nn

class GPT2Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)     # Lookup for token IDs
        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))  # Learned positions
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x is (B, T) – batch of token IDs
        tok_emb = self.token_embeddings(x)  # (B, T, D)
        pos_emb = self.positional_embeddings[:, :x.size(1), :]  # (1, T, D) -> broadcasted
        return self.dropout(tok_emb + pos_emb)  # (B, T, D)