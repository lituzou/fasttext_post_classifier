import torch


class ClassificationDataset:
    def __init__(self, titles, targets):
        self.titles = titles
        self.targets = targets
    
    def __len__(self):
        return len(self.titles)

    def __getitem__(self, index):
        word, bigram, trigram = self.titles[index]
        target = self.targets[index]
        return {
            "words": torch.tensor(word, dtype=torch.int),
            "bigrams": torch.tensor(bigram, dtype=torch.int),
            "trigrams": torch.tensor(trigram, dtype=torch.int),
            "targets": torch.tensor(target, dtype=torch.long)
        }