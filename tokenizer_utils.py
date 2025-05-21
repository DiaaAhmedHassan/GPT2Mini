class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word2idx = vocab
        self.idx2word = {idx: word for word, idx in vocab.items()}

    def encode(self, text):
        return [self.word2idx.get(w, self.word2idx['<unk>']) for w in text.split()]

    def decode(self, ids):
        return " ".join([self.idx2word.get(i, "<unk>") for i in ids])