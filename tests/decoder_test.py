import torch
import torch.nn as nn
import json

import sys
sys.dont_write_bytecode = True # disabling __pycache__
sys.path.insert(0, '../model/')
from decoder import DecoderLayer

# --- Load Vocab ---
with open("../check_points/tokenizer_vocab.json") as f:
    word2idx = json.load(f)
idx2word = {int(v): k for k, v in word2idx.items()}
pad_id = word2idx["<pad>"]

# --- Tokenizer Function ---
def simpleTokenizer(text):
    return [word2idx.get(word, word2idx['<unk>']) for word in text.split()]

# --- Dummy Config Class ---
class DummyConfig:
    def __init__(self):
        self.hidden_size = 64
        self.num_heads = 4
        self.ff_hidden_size = 256
        self.dropout = 0.1

# --- Step 1: Clean + Tokenize Input ---
sentence = "one day a little cat ran away"
tokens = simpleTokenizer(sentence)
print("Token IDs:", tokens)

# --- Step 2: Pad & Convert to Tensor ---
max_len = 20
tokens += [pad_id] * (max_len - len(tokens))
input_tensor = torch.tensor([tokens], dtype=torch.long)  # [1, seq_len]

# --- Step 3: Embedding ---
embedding = nn.Embedding(len(word2idx), 64, padding_idx=pad_id)
embedded_input = embedding(input_tensor)  # [1, seq_len, hidden_size]

# --- Step 4: Causal Mask ---
T = max_len
mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

# --- Step 5: Decoder Forward ---
decoder = DecoderLayer(DummyConfig())
output = decoder(embedded_input, mask)  # [1, seq_len, hidden_size]

# --- Step 6: Check Output ---
print("Output shape:", output.shape)
print("First token output vector:", output[0, 0])
