from model import FastText
import pandas as pd
import config
from sklearn import preprocessing, model_selection, metrics
import numpy as np
import engine
import dataset
import torch
from torch.utils.data import DataLoader
import utils


def run_training(model_path):
    df = pd.read_csv(config.DATA_FILE)
    targets = df['label'].to_list()
    titles = df['desc'].to_list()
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(df['label'])
    vocab_list = utils.build_vocab(titles)
    vocab_enc = preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=len(vocab_list), dtype=np.int)
    vocab_enc.fit(vocab_list.reshape(-1, 1))
    targets_enc = label_enc.transform(targets)
    titles_enc = list(
        map(lambda x: utils.encode_sentence(x, vocab_enc), titles))
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
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1)

    for epoch in range(config.EPOCH):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        test_preds, test_loss = engine.eval_fn(model, test_loader)
        test_preds = torch.argmax(test_preds, dim=1)
        test_preds = test_preds.detach().cpu().numpy()
        for label_class in label_enc.classes_:
            class_code = label_enc.transform([label_class])[0]
            class_acc = np.sum((test_targets == class_code) & (test_targets == test_preds)) / np.sum(test_targets == class_code)
            print(f'{label_class}: {class_acc}', end='\t')
        print()
        accuracy = metrics.accuracy_score(test_targets, test_preds)
        print(
            f'Epoch={epoch}, train loss={train_loss}, test_loss={test_loss}, accuracy={accuracy}')
        scheduler.step()

    if model_path is not None:
        torch.save(model.state_dict(), f"{model_path}.pth")
        np.save(f'{model_path}_label_enc.npy', label_enc.classes_)
        np.save(f'{model_path}_vocab_enc.npy', vocab_enc.categories_)


if __name__ == '__main__':
    run_training('model/post_classifier')
