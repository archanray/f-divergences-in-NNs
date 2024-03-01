import torch
from torch import nn
from src.model import LeNet5 as model
from src.dataset_file import MNIST_dataset as Dataset
from torchinfo import summary
from torchmetrics import Accuracy
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def main():
    # uncomment to check model summary
    # summary(
    #     model=model_lenet5v1, 
    #     input_size=(1, 1, 28, 28), 
    #     col_width=20, 
    #     col_names=['input_size', 'output_size', 'num_params', 'trainable'], 
    #     row_settings=['var_names'], 
    #     verbose=0)

    # Experiment tracking
    timestamp = datetime.now().strftime("%Y-%m-%d")
    experiment_name = "MNIST"
    model_name = "LeNet5"
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
    writer = SummaryWriter(log_dir)

    # device-agnostic setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_lenet5 = model.to(device)

    # set up loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_lenet5.parameters(), lr=0.001)
    accuracy = Accuracy(task='multiclass', num_classes=10)
    accuracy = accuracy.to(device)

    # set up the dataloaders
    train_dataloader, val_dataloader, test_dataloader = Dataset(BATCH_SIZE=32)

    EPOCHS = 12

    for epoch in tqdm(range(EPOCHS)):
        # Training loop
        train_loss, train_acc = 0.0, 0.0
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
            
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
            
        # Validation loop
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
            
        writer.add_scalars(
            main_tag="Loss", 
            tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, 
            global_step=epoch
            )
        writer.add_scalars(
            main_tag="Accuracy", 
            tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, 
            global_step=epoch)
        
        print(
            f"Epoch: {epoch}| 
            Train loss: {train_loss: .5f}| 
            Train acc: {train_acc: .5f}| 
            Val loss: {val_loss: .5f}| 
            Val acc: {val_acc: .5f}")
    
if __name__ == "__main__":
    main()