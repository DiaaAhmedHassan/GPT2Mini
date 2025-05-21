# data/dataset.py

import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, token_sequences, seq_len):
        self.seq_len = seq_len
        self.samples = []

        for tokens in token_sequences:
            for i in range(0, len(tokens) - seq_len):
                input_seq = tokens[i:i + seq_len]
                target_seq = tokens[i + 1:i + 1 + seq_len]
                self.samples.append((input_seq, target_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return torch.tensor(input_seq), torch.tensor(target_seq)