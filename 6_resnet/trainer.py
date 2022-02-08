import torch
import torch.nn as nn

def check_accuracy(loader, model, device):
    num_correct = 0
    num_sample = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)
            _, prediction = scores.max(1)
            num_correct += (prediction == y).sum()
            num_sample += prediction.size(0)
            
    model.train()
    acc = round(float(num_correct) / float(num_sample), 4)
    
    return acc


def train_epoch(dataloader, model, criterion, optimizer, device='cpu'):
    model = model.to(device)
    model.train()
    loss_epoch = []

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)

        # forward
        score = model(data)
        loss = criterion(score, target)
        loss_epoch.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # optimizer
        optimizer.step()
        
    train_loss = sum(loss_epoch) / len(loss_epoch)
    train_acc = check_accuracy(dataloader, model, device)

    return train_loss, train_acc

def valid_epoch(validloader, model, criterion, device='cpu'):
    model = model.to(device)
    model.eval()
    loss_epoch = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validloader):
            data = data.to(device)
            target = target.to(device)

            score = model(data)
            loss = criterion(score, target)
            loss_epoch.append(loss.item())

        valid_loss = sum(loss_epoch) / len(loss_epoch)
        valid_acc = check_accuracy(validloader, model, device)

    return valid_loss, valid_acc