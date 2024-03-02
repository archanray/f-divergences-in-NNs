import torch
from torch import nn
from src.model import LeNet5 as model
from src.dataset_file import MNIST_dataset as Dataset
from torchmetrics import Accuracy
from tqdm import tqdm
import os
import argparse
from utils.runner import train, validate
from utils.utilities import tensorWriter

def main(args):
    # Experiment tracker
    writer = tensorWriter(experiment_name = "MNIST", model_name = "LeNet5", dir="runs")

    # device-agnostic setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_lenet5 = model().to(device)

    # set up loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_lenet5.parameters(), lr=0.0005)
    accuracy = Accuracy(task='multiclass', num_classes=10)
    accuracy = accuracy.to(device)

    # set up the dataloaders
    train_dataloader, val_dataloader, test_dataloader = Dataset(BATCH_SIZE=args.BATCH_SIZE)

    EPOCHS = args.EPOCHS

    for epoch in tqdm(range(EPOCHS)):
        # Training loop
        train_loss, train_acc = 0.0, 0.0
        # train model
        train_loss, train_acc = train(
                                    train_dataloader=train_dataloader,
                                    model_lenet5=model_lenet5,
                                    loss_fn=loss_fn,
                                    optimizer=optimizer,
                                    accuracy=accuracy,
                                    device=device,
                                    train_loss=0,
                                    train_acc=0
                                    )
            
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        # validate model    
        val_loss, val_acc = validate(
                                    model_lenet5=model_lenet5, 
                                    val_dataloader=val_dataloader, 
                                    loss_fn=loss_fn, 
                                    accuracy=accuracy, 
                                    device=device
                                    )
        
        # writer in the tensorboard    
        writer.add_scalars(
            main_tag="Loss", 
            tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, 
            global_step=epoch
            )
        writer.add_scalars(
            main_tag="Accuracy", 
            tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, 
            global_step=epoch)
        
        print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}")
    writer.close()
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Handle some inputs.')
    parser.add_argument('--batchsize', '-s', dest='BATCH_SIZE', type=int,
                        default=32,
                        help='indicate batchsize')
    parser.add_argument('--epochs', '-e', dest='EPOCHS', type=int,
                        default=20,
                        help='EPOCHS to train the model')

    args = parser.parse_args()
    main(args)
