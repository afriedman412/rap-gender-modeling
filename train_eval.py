import torch
import torch.nn as nn
import numpy as np

def train_epoch(
    model, data_loader, criterion, optimizer, device, scheduler, n_examples
):
    model = model.train()
    losses = []
    correct_preds = 0

    for dl in data_loader:
        input_ids = dl['input_ids'].to(device)
        attention_mask = dl['attention_mask'].to(device)
        targets = dl['targets'].to(device)

        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, targets)

        correct_preds += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_preds.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, criterion, device, n_examples):
    model = model.eval()

    losses = []
    correct_preds = 0

    with torch.no_grad():
        for dl in data_loader:
            input_ids = dl['input_ids'].to(device)
            attention_mask = dl['attention_mask'].to(device)
            targets = dl['targets'].to(device)

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, targets)
            correct_preds += torch.sum(preds==targets)
            losses.append(loss.item())

    return correct_preds.double() / n_examples, np.mean(losses)
