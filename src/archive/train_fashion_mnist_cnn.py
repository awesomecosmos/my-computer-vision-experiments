import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.nn.functional as F 
import pickle

import utils

my_params = {
    'n_epochs':10,
    # number of output channels for each convolution
    # good rule of thumb is to double every time
    'conv1_out_channels':32,
    'conv2_out_channels':64,
    'conv3_out_channels':128,
    # number of fully-connected layers - max 4
    'fc1_in_features':128 * 25 * 25, # x * y * y, where x=conv3_out_channels and y=dim(conv3_out_channels)
    'fc1_out_features':256, # perhaps we can halve every time now?
    'fc2_out_features':128,
    'fc3_out_features':64,
    # kernel size - same for all convolutions and for all poolings
    'conv_kernel_size':2,
    'pool_kernel_size':1,
    # stride - same for all convolutions and for all poolings
    'conv_stride':1,
    'pool_stride':1,
    # desired pooling type - either of {'max', 'avg'}
    'pool_type': 'max',
    # desired activation - either of {'relu', 'sigmoid', 'tanh'}
    'activation_type': 'relu',
    # optimizer - either of {'sgd', 'adam'}
    'optimizer':'sgd'
}

device = torch.device("cuda")
print(f"Using {device} device")

transform = transforms.Compose(
    [transforms.ToTensor(),
    # transforms.Normalize((1,), (1,))
    ])

train_data = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
test_data = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# experiment - play with the number of batches I want
# default is 4
batch_size = 5

# Create data loaders for our datasets; shuffle for training, not for test
training_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Report split sizes
print('Training set has {} instances'.format(len(train_data)))
print('Testing set has {} instances'.format(len(test_data)))



# instantiating the model
fashion_classifier_model = utils.MyFashionClassifier(my_params)

loss_fn = torch.nn.CrossEntropyLoss() 

my_optimizer = utils.choose_optimizer(
    model=fashion_classifier_model, 
    optimizer_choice=my_params['optimizer'], 
    learning_rate=0.001,
    momentum=0.9
    )

def train_model(
        model, params, optimizer, loss_fn, training_loader, num_epochs=10
    ):
    
    losses = []
    accuracies = []

    # iterating over number of epochs
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_samples = 0

        for i, (inputs, labels) in enumerate(training_loader):
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            predicted = torch.argmax(outputs, 1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_samples += labels.size(0)

            if (i + 1) % 1000 == 0:
                batch_loss = epoch_loss / (i + 1)
                losses.append(batch_loss)

                accuracy = 100 * epoch_correct / epoch_samples
                accuracies.append(accuracy)

                print(f'Epoch [{epoch+1}/{num_epochs}], Batch {i+1}, Loss: {batch_loss:.4f}, Accuracy: {accuracy:.4f}')

        # End of epoch logging
        epoch_loss_avg = epoch_loss / len(training_loader)
        losses.append(epoch_loss_avg)
        accuracy = 100 * epoch_correct / epoch_samples
        accuracies.append(accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss_avg:.4f}, Accuracy: {accuracy:.4f}')
        
        utils.plot_loss_accuracy_curves(losses, accuracies, epoch+1)

    return losses, accuracies

losses, accuracies = train_model(
    model=fashion_classifier_model, 
    params=my_params, 
    optimizer=my_optimizer, 
    loss_fn=loss_fn, 
    training_loader=training_loader,
    num_epochs=my_params['n_epochs']
    )

# Save the trained model's parameters
torch.save(fashion_classifier_model.state_dict(), '../models/fashion_mnist_cnn.pth')
with open('../models/tuple_data.pkl', 'wb') as file:
    pickle.dump((losses, accuracies), file)