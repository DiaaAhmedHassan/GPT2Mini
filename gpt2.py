# model/gpt2.py (add this later)

import torch
import torch.nn as nn
from decoder import DecoderLayer
from embeddings import GPT2Embedding

class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = GPT2Embedding(config.vocab_size, config.hidden_size, config.max_seq_len)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x, mask=None):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)
