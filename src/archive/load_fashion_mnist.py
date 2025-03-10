import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.nn.functional as F 

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def load_fashion_mnist():

    # defining transforms
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

    # defining data for train and test
    train_data = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    test_data = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    # setting correct device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    batch_size = 5
    print(f"batch_size={batch_size}")
    # Create data loaders for our datasets; shuffle for training, not for test
    training_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Class labels
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Report split sizes
    print('Training set has {} instances'.format(len(train_data)))
    print('Testing set has {} instances'.format(len(test_data)))

    return (training_loader, test_loader, classes)


def test_load_fashion_mnist(training_loader, classes, batch_size):
    dataiter = iter(training_loader)
    images, labels = next(dataiter)

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    utils.matplotlib_imshow(img_grid, one_channel=True)
    print('  '.join(classes[labels[j]] for j in range(batch_size)))

if __name__ == '__main__':
    load_fashion_mnist()
