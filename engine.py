from tqdm import tqdm
import torch
import config

def train_fn(model, data_loader, optimizer):
    model.train()
    final_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))
    for data in tk0:
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(model, data_loader):
    model.eval()
    final_loss = 0
    final_preds = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for data in tk0:
            for key, value in data.items():
                data[key] = value.to(config.DEVICE)
            batch_preds, loss = model(**data)
            final_loss += loss.item()
            final_preds.append(batch_preds)
        final_preds = torch.cat(final_preds, dim=0)
        return final_preds, final_loss / len(data_loader)
