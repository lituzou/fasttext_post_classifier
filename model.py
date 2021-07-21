import torch
import torch.nn as nn
import torch.nn.functional as func
import config
from sklearn import preprocessing
import numpy as np


class FastText(nn.Module):
    def __init__(self, num_classes, num_vocab, padding_enc):
        super(FastText, self).__init__()
        char_dim = 300
        self.embedding1 = nn.Embedding(
            num_vocab, char_dim, padding_idx=padding_enc)
        self.embedding2 = nn.Embedding(config.NGRAM_VOCAB_SIZE, char_dim)
        self.embedding3 = nn.Embedding(config.NGRAM_VOCAB_SIZE, char_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(char_dim * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, words, bigrams, trigrams, targets=None):
        em_words = self.embedding1(words)
        em_bigrams = self.embedding2(bigrams)
        em_trigrams = self.embedding3(trigrams)
        x = torch.cat([em_words, em_bigrams, em_trigrams], -1)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)

        if targets is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(x, targets)
            return x, loss

        return x, None


if __name__ == '__main__':
    vocab_list = np.array(
        [c for c in 'abcdefghijklmnopqrstuvwxyz'] + [config.PAD])
    enc = preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=len(vocab_list), dtype=np.int)
    enc.fit(vocab_list.reshape(-1, 1))
    print('Vocab size', len(enc.categories_[0])+1)
    batch_sz = 5
    words = torch.randint(0, len(vocab_list), (batch_sz, config.PAD_SIZE))
    bigrams = torch.randint(0, config.NGRAM_VOCAB_SIZE-1,
                            (batch_sz, config.PAD_SIZE))
    trigrams = torch.randint(
        0, config.NGRAM_VOCAB_SIZE-1, (batch_sz, config.PAD_SIZE))
    num_classes = 10
    targets = torch.randint(0, num_classes-1, (batch_sz,))
    padding_enc = int(enc.transform([[config.PAD]]).item())
    model = FastText(num_classes, len(enc.categories_[0])+1, padding_enc)
    result, loss = model(words=words, bigrams=bigrams,
                         trigrams=trigrams, targets=targets)
    print(result.size(), loss)
