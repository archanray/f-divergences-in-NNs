import torch
from torchvision import transforms
from torchvision.datasets import mnist
from torch.utils.data import DataLoader

def MNIST_dataset(BATCH_SIZE=256):
    # load data
    train_val_dataset = mnist.MNIST(root="./datasets/mnist/train", train=True, download=True, transform=transforms.ToTensor())
    test_dataset = mnist.MNIST(root="./datasets/mnist/test", train=False, download=True, transform=transforms.ToTensor())

    # Calculate mean and std
    imgs = torch.stack([img for img, _ in train_val_dataset], dim=0)
    mean = imgs.view(1, -1).mean(dim=1)    # or imgs.mean()
    std = imgs.view(1, -1).std(dim=1)     # or imgs.std()
    # prepare transforms
    mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    # apply transforms
    train_val_dataset = mnist.MNIST(root="./datasets/mnist/train", train=True, download=False, transform=mnist_transforms)
    test_dataset = mnist.MNIST(root="./datasets/mnist/test", train=False, download=False, transform=mnist_transforms)
    
    # split train to test and validation
    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])
    
    # set up the dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_dataloader, val_dataloader, test_dataloader