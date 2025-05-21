import torch
import torch.nn as nn

class GPT2Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_emb = self.token_embeddings(x)  # [B, T, D]
        pos_emb = self.positional_embeddings[:, :seq_len, :]
        return self.dropout(tok_emb + pos_emb)