from model import FastText
import pandas as pd
import config
from sklearn import preprocessing, model_selection, metrics
import numpy as np
import engine
import dataset
import torch
from torch.utils.data import DataLoader


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


def run_training(model_path):
    df = pd.read_csv(config.DATA_FILE)
    targets = df['label'].to_list()
    titles = df['desc'].to_list()
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(df['label'])
    vocab_list = build_vocab(titles)
    vocab_enc = preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=len(vocab_list), dtype=np.int)
    vocab_enc.fit(vocab_list.reshape(-1, 1))
    targets_enc = label_enc.transform(targets)
    titles_enc = list(map(lambda x: encode_sentence(x, vocab_enc), titles))
    train_titles, test_titles, train_targets, test_targets = model_selection.train_test_split(
        titles_enc, targets_enc, test_size=0.1, random_state=42
    )

    train_dataset = dataset.ClassificationDataset(
        titles=train_titles, targets=train_targets)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_WORKERS, shuffle=True, pin_memory=True)

    test_dataset = dataset.ClassificationDataset(
        titles=test_titles, targets=test_targets)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_WORKERS, shuffle=False, pin_memory=True)

    padding_enc = int(vocab_enc.transform([[config.PAD]]).item())
    model = FastText(
        num_classes=len(label_enc.classes_),
        num_vocab=len(vocab_enc.categories_[0])+1,
        padding_enc=padding_enc
    )
    model = model.to(config.DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    for epoch in range(config.EPOCH):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        test_preds, test_loss = engine.eval_fn(model, test_loader)
        test_preds = torch.argmax(test_preds, dim=1)
        test_preds = test_preds.detach().cpu().numpy()
        for i, item in enumerate(zip(test_preds, test_targets)):
            if i < 10:
                print(item, end=',')
            else:
                print()
                break
        accuracy = metrics.accuracy_score(test_targets, test_preds)
        print(f'Epoch={epoch}, train loss={train_loss}, test_loss={test_loss}, accuracy={accuracy}')
        scheduler.step()        

    if model_path is not None:
        torch.save(model.state_dict(), f"{model_path}.pth")
        np.save(f'{model_path}_label_enc.npy', label_enc.classes_)
        np.save(f'{model_path}_vocab_enc.npy', vocab_enc.categories_)
    

if __name__ == '__main__':
    run_training('model/post_classifier')
