import torch

DATA_FILE = 'post_data.csv'
BATCH_SIZE = 8
NUM_WORKERS = 2
EPOCH = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_VOCAB_SIZE = 2000
UNK = '<UNK>'
PAD = '<PAD>'
PAD_SIZE = 80
NGRAM_VOCAB_SIZE = 250499