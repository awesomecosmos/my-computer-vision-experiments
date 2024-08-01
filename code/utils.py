import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 

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

def plot_loss_accuracy_curves(losses, accuracies, epoch_num):
    """Function to plot loss-accuracy curve.

    Args:
        losses (list): List of losses.
        accuracies (list): List of accuracies.
        epoch_num (int): Epoch number, for plot title.
    """
    plt.figure(figsize=(12, 4))
    # plotting loss vs epoch
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title(f'Epoch {epoch_num}: Loss Curve')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    # plotting accuracy vs epoch
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title(f'Epoch {epoch_num}: Accuracy Curve (avg={np.round(np.mean(accuracies),2)}%)')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy (%)')
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