import torch
from tqdm.notebook import tqdm

def train(train_dataloader=None, model_lenet5=None, loss_fn=None, optimizer=None, accuracy=None, device=None, train_loss=0, train_acc=0):
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        
        model_lenet5.train()
        
        y_pred = model_lenet5(X)
        
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        acc = accuracy(y_pred, y)
        train_acc += acc
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    # return train_loss, train_acc, model_lenet5, loss, optimizer
    return train_loss, train_acc

def validate(model_lenet5=None, val_dataloader=None, loss_fn=None, accuracy=None, device=None):
    val_loss, val_acc = 0.0, 0.0
    model_lenet5.eval()
    with torch.inference_mode():
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            
            y_pred = model_lenet5(X)
            
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
            
            acc = accuracy(y_pred, y)
            val_acc += acc
            
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)
    return val_loss, val_acc