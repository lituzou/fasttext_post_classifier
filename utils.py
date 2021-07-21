import config
from sklearn import preprocessing
import numpy as np


def build_vocab(sentences):
    vocab = dict()
    for s in sentences:
        for c in s:
            vocab[c] = vocab.get(c, 0) + 1
    vocab_list = sorted(list(vocab.keys()), key=lambda k: vocab[k], reverse=True)[
        :config.MAX_VOCAB_SIZE-2]
    vocab_list.append(config.PAD)
    return np.array(vocab_list)


def bigram_gen(word, buckets):
    t1 = np.roll(word, 1)
    t1[0] = 0
    return (t1 * 14918087) % buckets


def trigram_gen(word, buckets):
    t1 = bigram_gen(word, buckets)
    t2 = np.roll(word, 2)
    t2[:2] = 0
    return (((((t2 * 14918087) % buckets) * 18408749) % buckets) + t1) % buckets


def encode_sentence(sentence: str, vocab_enc: preprocessing.OrdinalEncoder):
    # Padding
    token = [c for c in sentence]
    if len(token) < config.PAD_SIZE:
        token.extend([config.PAD] * (config.PAD_SIZE - len(token)))
    else:
        token = token[:config.PAD_SIZE]
    token = np.array(token)
    # Encoding
    word = vocab_enc.transform(token.reshape(-1, 1)).reshape(-1)
    bigram = bigram_gen(word, config.NGRAM_VOCAB_SIZE)
    trigram = trigram_gen(word, config.NGRAM_VOCAB_SIZE)
    return word, bigram, trigram
