import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader
import time
import os
import json
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging

import utils

################################################################################


def train_model(model, train_loader, criterion, hyperparameters):
    saveDirResults = f"../data/results/{hyperparameters['model_name']}_pretrained{hyperparameters['pretrained_model']}.pkl"
    losses, accuracies = [], []
    model = model.to(device)
    model.train()

    for epoch in range(hyperparameters['n_epochs']):
        running_loss = 0.0
        correct, total = 0, 0  # Track accuracy
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if hyperparameters['optimizer'] == 'Adam':
                optimizer = optim.Adam(model.fc.parameters(), lr=hyperparameters['lr'])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total  # Accuracy calculation
        
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        logger.info(f"Epoch {epoch+1}/{hyperparameters['n_epochs']}, Loss: {epoch_loss:.4f}, Accuracy: {100*epoch_accuracy:.4f}%")
    
    # saving numerical results
    numerical_results = {
            "losses": losses,
            "accuracies": accuracies
        }
    
    utils.plot_loss_accuracy_curves(
        losses, accuracies, 
        saveTag=f"train_{hyperparameters['model_name']}_pretrained{hyperparameters['pretrained_model']}", 
        figSaveDir=f"../figs/pretrained_experiments/"
        )

def test_model(model, test_loader):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.info(f"Test Accuracy: {100 * correct / total:.2f}%")
    # add cm
    
def run_pretrained_model_experiment(train_loader, test_loader, hyperparameters):
    if hyperparameters['model_name'] == 'ResNet50':
        if hyperparameters['pretrained_model'] is True:
            model = resnet50(weights='DEFAULT')
        else:   
            model = resnet50(weights=None)
    model = model.to(device)

    # Modify the final fully connected layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)  

    # Freeze all layers except the final classifier
    for param in model.parameters():
        param.requires_grad = False  # Freeze convolutional layers
    for param in model.fc.parameters():
        param.requires_grad = True  # Unfreeze final layer

    criterion = nn.CrossEntropyLoss()

    logger.info(f"Training model: ({hyperparameters['model_name']}, pretrained={hyperparameters['pretrained_model']})")
    train_model(model, train_loader, criterion, hyperparameters)
    test_model(model, test_loader)

    return model

################################################################################

# setting up logging file
logger = utils.setup_logger('../logs/pretrained_models_logs.txt')

# Log script start time
start_time = time.time()
start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger.info("=" * 50)
logger.info(f"Running script: {os.path.basename(__file__)}")
logger.info(f"Script started at: {start_timestamp}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')
if (torch.cuda.is_available()):
    print('Number of CUDA Devices:', torch.cuda.device_count())
    print('CUDA Device Name:',torch.cuda.get_device_name(0))
    logger.info(f'CUDA Device Total Memory [GB]:{torch.cuda.get_device_properties(0).total_memory/1e9}')

# Transform: Resize to 224x224 (ResNet expects this) and repeat channels
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1]
])

# Load dataset
train_dataset = torchvision.datasets.FashionMNIST(root='../data/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='../data/', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

# setting experiment hyperparameters
lst_of_models = ['ResNet50']
pretrained = [False, True]

for model_name in lst_of_models:
    for flag in pretrained:
        logger.info("Starting new experiment!")   
        hyperparameters = {
            'model_name':model_name,
            'pretrained_model':flag,
            'batchSize':64,
            'n_epochs':10,
            'optimizer':'Adam',
            'lr':0.001
        }
        logger.info("Hyperparameters:")
        logger.info(json.dumps(hyperparameters, indent=4))  # Pretty-print dictionary
        model = run_pretrained_model_experiment(train_loader, test_loader, hyperparameters)

        # plotting
        sample_image, _ = test_dataset[1]  # Get the first test image
        figSaveTag = f'{hyperparameters['model_name']}-pretrained{hyperparameters['pretrained_model']}'
        utils.visualize_feature_maps(model, device, sample_image, figSaveTag=figSaveTag, figSaveDir="../figs/pretrained_experiments", layer_name="layer1")  # Change "layer1" to "layer2", "layer3" for deeper layers
        utils.plot_confusion_matrix(model, device, test_loader, class_names=train_dataset.classes, figSaveTag=figSaveTag, figSaveDir="../figs/pretrained_experiments")

# Log script end time
end_time = time.time()                                                                              
end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
elapsed_time = end_time - start_time
logger.info(f"Script ended at: {end_timestamp}")
logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
logger.info("=" * 50)