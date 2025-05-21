class Tokenizer:
    def __init__(self):
        self.vocab = dict()
        self.word2idx = dict()
        self.idx2word = dict()

    def upload_texts(self, texts: list[str]):
        words = set()
        
        for text in texts:
            words.update(text.split())

        self.vocab = {word: i for i, word in enumerate(sorted(words), start=0)}
        self.word2idx = self.vocab
        self.idx2word = {i: word for word, i in self.vocab.items()}

    def upload_vocab(self, vocab: dict[str, int]): 
        self.vocab = vocab
        self.word2idx = vocab
        self.idx2word = {i: word for word, i in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        # dict.get(key, default_value)
        # self.word2idx.get(w, self.word2idx['<unk>'])
        return [self.word2idx[w] for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        # self.idx2word.get(i, "<unk>")
        return " ".join([self.idx2word[i] for i in ids])
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)

# remove non alphabetical character
import re
import json
def clean_text(txt):
    txt = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip().lower()