
"""CIFAR10 dataset."""

import torch
import torchvision
import torchvision.transforms as transforms

def prepare_data(cfg):
    transform=transforms.Compose([
       torchvision.transforms.ToTensor(),
       torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg["TRAIN"]["BATCH_SIZE"], 
                                               shuffle=True, num_workers=cfg["DATA_LOADER"]["NUM_WORKERS"])

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg["TEST"]["BATCH_SIZE"], 
                                              shuffle=False, num_workers=cfg["DATA_LOADER"]["NUM_WORKERS"])
    
    return train_loader, test_loader