import torch
from torchvision import datasets, transforms
import numpy as np
from opacus import PrivacyEngine
from tqdm import tqdm

# Training a simple PyTorch classification model

# Use the built-in MNIST dataset from PyTorch

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../mnist",
        train=True,  # this is training data
        download=True,  # download data
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # converts image or array to a Tensor
                transforms.Normalize((0.1307,), (0.3081,)),
            ]  # 0.1307 is the mean and 0.3081 is the standard deviation
        ),
    ),
    batch_size=64,
    shuffle=True,
    num_workers=1,  # each time an iterator of a DataLoader is created, a worker process is created, and the worker is used to initialize and fetch data
    pin_memory=True,  # enables automatic memory pinning, enabling fast data transfer
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../mnist",
        train=False,  # this is test data
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # converts image or array to a Tensor
                transforms.Normalize((0.1307,), (0.3081,)),
            ]  # 0.1307 is the mean and 0.3081 is the standard deviation
        ),
    ),
    batch_size=1024,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

# This Sequential PyTorch neural network model predicts the label of images from the MNIST dataset
# The network has 2 convolutional layers, 2 fully connected layers (along with ReLU layers to set any negative values to zero and maxpooling layers to reduce size of features),
# The first layer accepts images of size (28 x 28) as input, the final returns a vector of probabilities of size (1 x 10), predicting the digit each image represents

# A Stochastic Gradient Descent (SGD) optimizer improves the model, and the learning rate is set to 0.05

model = torch.nn.Sequential(
    torch.nn.Conv2d(
        1, 16, 8, 2, padding=3
    ),  # applies 2D convolution over an input signal composed of multiple input planes. There is one in channel, 16 out channels, a kernel size of 8, stride of 2, and padding of 3
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(
        2, 1
    ),  # applies a 2D max pooling over an input signal composed of several input planes, kernel size is 2, stride is 1
    torch.nn.Conv2d(16, 32, 4, 2),
    # 16 input, 32 output, kernel is 4, stride is 2, no padding
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 1),
    torch.nn.Flatten(),  # flattens input by reshaping it into a 1-dimensional tensor
    torch.nn.Linear(
        512, 32
    ),  # applies an affine linear transformation (y=xA^t + b) to the incoming data. input has a size of 32 * 4 * 4 (512) and output has a size of 32
    torch.nn.ReLU(),
    torch.nn.Linear(
        32, 10
    ),  # for this linear transformation, input has a size of 32, output has a size of 10
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
# applies stochastic gradient descent

# Source for example: https://openmined.org/blog/differentially-private-deep-learning-using-opacus-in-20-lines-of-code/
