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

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def cnn_dims(input_matrix_size, kernel_size):
    """Function to obtain dimensions of feature map post-convolution.

    Args:
        input_matrix_size (int): m in mxm size of input matrix.
        kernel_size (int): f in fxf size of convolution kernel.

    Returns:
        output_size_n (int): n in nxn size of convolved matrix.
    """
    output_size_n = input_matrix_size - kernel_size + 1
    print(f'Input matrix size: {input_matrix_size}x{input_matrix_size}')
    print(f'Kernel size: {kernel_size}x{kernel_size}')
    print(f'Therefore output convolved matrix size: {output_size_n}x{output_size_n}')
    return output_size_n

def pooling_dims(input_matrix_size, kernel_size, stride):
    """Function to obtain dimensions of feature map post-pooling.

    Args:
        input_matrix_size (int): m in mxm size of input matrix.
        kernel_size (int): f in fxf size of pooling kernel.
        stride (int): stride for pooling.

    Returns:
        output_size_n (int): n in nxn size of pooled matrix.
    """
    output_size_n = int(np.floor((input_matrix_size - kernel_size) / stride) + 1)
    print(f'Input matrix size: {input_matrix_size}x{input_matrix_size}')
    print(f'Kernel size: {kernel_size}x{kernel_size} + stride: {stride}')
    print(f'Therefore output pooled matrix size: {output_size_n}x{output_size_n}')
    return output_size_n

def choose_optimizer(model, optimizer_choice, learning_rate=0.01, momentum=None):
    """Function to choose and initialize optimizer for back-propagation.

    Args:
        model: model.
        optimizer_choice (str): Type of optimizer. Either of {'sgd', 'adam'}.
        learning_rate (float): Desired learning rate. Defaults to 0.01.
        momentum (float, optional): Desired momentum. Defaults to None.

    Returns:
        optimizer: Instantiated optimizer.
    """
    if optimizer_choice == 'sgd': 
        optimizer = torch.optim.SGD(
            params=model.parameters(), 
            lr=learning_rate, 
            momentum=momentum
            ) 
    elif optimizer_choice=='adam': 
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    return optimizer

def plot_loss_accuracy_curves(losses, accuracies, saveTag, figSaveDir):
    """Function to plot loss-accuracy curve.

    Args:
        losses (list): List of losses.
        accuracies (list): List of accuracies.
        saveTag (str): Used in title and savename of plot.
        figSaveDir (str): Directory to save figure in.
    """
    plt.figure(figsize=(12, 4))
    plt.suptitle(saveTag, fontsize=14, fontweight='bold')
    # Plot loss vs epoch
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title(f'Epoch vs Loss (avg={np.round(np.mean(losses),4)})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # Plot accuracy vs epoch
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title(f'Epoch vs Accuracy (avg={np.round(np.mean(accuracies),2)}%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(f'{figSaveDir}/loss_acc_{saveTag}.jpg', dpi=300)
    plt.show()


class MyFashionClassifier(nn.Module):

    def __init__(self, my_params):
        super(MyFashionClassifier, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=my_params['conv1_out_channels'],
            kernel_size=my_params['conv_kernel_size'],
            stride=my_params['conv_stride']
        )
        self.conv2 = nn.Conv2d(
            in_channels=my_params['conv1_out_channels'],
            out_channels=my_params['conv2_out_channels'],
            kernel_size=my_params['conv_kernel_size'],
            stride=my_params['conv_stride']
        )
        self.conv3 = nn.Conv2d(
            in_channels=my_params['conv2_out_channels'],
            out_channels=my_params['conv3_out_channels'],
            kernel_size=my_params['conv_kernel_size'],
            stride=my_params['conv_stride']
        )
        
        self.pool_type = my_params['pool_type']
        if self.pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=my_params['pool_kernel_size'], stride=my_params['pool_stride'])
        elif self.pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=my_params['pool_kernel_size'], stride=my_params['pool_stride'])

        self.activation_type = my_params['activation_type']

        self.fc1 = nn.Linear(
            in_features=my_params['fc1_in_features'],
            out_features=my_params['fc1_out_features']
        )
        self.fc2 = nn.Linear(
            in_features=my_params['fc1_out_features'],
            out_features=10
        )

    def forward(self, x):
        # if self.pool_type == 'max':
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 25 * 25)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def setup_logger(log_filename="logfile.txt"):
    # Create logger
    logger = logging.getLogger("CustomLogger")
    logger.setLevel(logging.INFO)

    # Create file handler (append mode)
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.INFO)

    # Create stream handler (console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    # Define log format
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def plot_confusion_matrix(model, device, test_loader, class_names, figSaveTag, figSaveDir):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {figSaveTag}")
    plt.savefig(f"{figSaveDir}/cm_{figSaveTag}.jpg",dpi=300)
    plt.show()

def visualize_feature_maps(model, device, image, figSaveTag, figSaveDir, layer_name="layer1"):
    model.eval()
    activation = None

    def hook_fn(module, input, output):
        nonlocal activation
        activation = output.detach()

    # Ensure layer exists
    named_modules = dict(model.named_modules())
    if layer_name not in named_modules:
        raise ValueError(f"Layer '{layer_name}' not found. Available layers: {list(named_modules.keys())}")

    # Register hook on chosen layer
    # layer = dict(model.named_modules())[layer_name]
    layer = named_modules[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    # Process a single image
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    _ = model(image)  # Forward pass to trigger hook

    hook.remove()

    # Plot feature maps
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))  # Show first 16 feature maps
    for i, ax in enumerate(axes.flat):
        if i >= activation.shape[1]: break  # Stop if fewer channels
        ax.imshow(activation[0, i].cpu().numpy(), cmap="viridis")
        ax.axis("off")
    plt.suptitle(f"Feature Maps from {layer_name}: {figSaveTag}")
    plt.savefig(f"{figSaveDir}/fm_{layer_name}_{figSaveTag}.jpg",dpi=900)
    plt.show()

def collate_results_into_pandas(results_list, hyperparameters, train_numerical_results, test_numerical_results):
    # Extract training results
    avg_train_loss = np.mean(train_numerical_results['losses'])
    avg_train_acc = np.mean(train_numerical_results['accuracies'])
    test_acc = test_numerical_results

    # Store in results list
    results_list.append({
        'Model': hyperparameters['model_name'],
        'Pretrained': hyperparameters['pretrained_model'],
        'Finetuning': hyperparameters['finetuning'],
        'Optimizer': hyperparameters['optimizer'],
        'Learning Rate': hyperparameters['lr'],
        'Momentum': hyperparameters['momentum'],
        'Weight Decay': hyperparameters['weight_decay'],
        'Batch Size': hyperparameters['batchSize'],
        'Epochs': hyperparameters['n_epochs'],
        'Avg Train Loss': avg_train_loss,
        'Avg Train Accuracy': avg_train_acc,
        'Test Accuracy': test_acc
    })
    return results_list

# def plot_random_predictions(model, test_loader, device, class_names, figSaveTag, figSaveDir, num_images=10):
#     """Plots random test images with their actual and predicted labels.

#     Args:
#         model (torch.nn.Module): Trained model.
#         test_loader (torch.utils.data.DataLoader): Dataloader for testing.
#         device (torch.device): Device (CPU/GPU).
#         class_names (list): List of class names.
#         num_images (int): Number of images to display.
#     """
#     model.eval()
#     images, labels = next(iter(test_loader))
#     images, labels = images.to(device), labels.to(device)

#     # Select random indices
#     indices = torch.randperm(images.size(0))[:num_images]
#     images, labels = images[indices], labels[indices]

#     with torch.no_grad():
#         outputs = model(images)
#         _, preds = torch.max(outputs, 1)  # Get predicted classes

#     # Plot images
#     fig, axes = plt.subplots(2, 5, figsize=(12, 5))
#     axes = axes.flatten()
#     for i in range(num_images):
#         img = images[i].cpu().squeeze().permute(1, 2, 0)  # Move channels to last dimension
#         axes[i].imshow(img, cmap='gray')
#         axes[i].set_title(f"Actual: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}", fontsize=10)
#         axes[i].axis('off')

#     plt.tight_layout()
#     # plt.title(f"Actual vs Predicted Labels: {figSaveTag}")
#     plt.savefig(f"{figSaveDir}/actual_vs_pred_examples_{figSaveTag}.jpg",dpi=300)
def plot_random_predictions(model, test_loader, device, class_names, figSaveTag, figSaveDir, num_images=10):
    """Plots random test images with their actual and predicted labels after unnormalizing.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (torch.utils.data.DataLoader): Dataloader for testing.
        device (torch.device): Device (CPU/GPU).
        class_names (list): List of class names.
        figSaveTag (str): Tag for saving the figure.
        figSaveDir (str): Directory to save the figure.
        num_images (int): Number of images to display.
    """
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Select random indices
    indices = torch.randperm(images.size(0))[:num_images]
    images, labels = images[indices], labels[indices]

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)  # Get predicted classes

    # Unnormalize images
    images = images * 0.5 + 0.5  # Reverse normalization

    # Plot images
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    for i in range(num_images):
        img = images[i].cpu().permute(1, 2, 0).numpy()  # Convert to NumPy and move channels

        axes[i].imshow(img)  # No need for cmap='gray' since we restored RGB
        axes[i].set_title(f"Actual: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{figSaveDir}/actual_vs_pred_examples_{figSaveTag}.jpg", dpi=300)
