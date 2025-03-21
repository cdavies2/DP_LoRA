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
    num_workers=1,  # each time an iterator of a DataLoader is created, a worker process is created
    # the worker is used to initialize and fetch data
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
# The network has 2 convolutional layers, and 2 fully connected layers
# (along with ReLU layers to set any negative values to zero and maxpooling layers to reduce size of features),
# The first layer accepts images of size (28 x 28) as input,
# the final returns a vector of probabilities of size (1 x 10),
#  predicting the digit each image represents

# A Stochastic Gradient Descent (SGD) optimizer improves the model, and the learning rate is set to 0.05

model = torch.nn.Sequential(
    torch.nn.Conv2d(
        1, 16, 8, 2, padding=3
    ),  # applies 2D convolution over an input signal composed of multiple input planes.
    # There is one in channel, 16 out channels, a kernel size of 8, stride of 2, and padding of 3
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(
        2, 1
    ),  # applies a 2D max pooling over an input signal composed of several input planes
    # kernel size is 2, stride is 1
    torch.nn.Conv2d(16, 32, 4, 2),
    # 16 input, 32 output, kernel is 4, stride is 2, no padding
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 1),
    torch.nn.Flatten(),  # flattens input by reshaping it into a 1-dimensional tensor
    torch.nn.Linear(
        512, 32
    ),  # applies an affine linear transformation (y=xA^t + b) to the incoming data.
    # input has a size of 32 * 4 * 4 (512) and output has a size of 32
    torch.nn.ReLU(),
    torch.nn.Linear(
        32, 10
    ),  # for this linear transformation, input has a size of 32, output has a size of 10
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
# applies stochastic gradient descent


# Attaching a PrivacyEngine makes our model Differentially Private
# Said engine also helps us track our privacy budget

# sample_size is the size of the sample set
# noise_multiplier is the ratio of the standard deviation
# of Gaussian noise distribution to L2 sensitivity of the function

# Global sensitivity is the maximum difference in output when one change is made to a dataset
# L2 is the square root of the sum of the squares of a vector, L1 is the sum of the vector's elements

# max_grad_norm is the value of the upper bound of the L2 norm of the loss gradients

privacy_engine = PrivacyEngine(secure_mode=False)


model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.3,
    max_grad_norm=1.0,
)

# When training a model, we iterate over the training batches of data
# and make the model predict the data label.
# Based on the prediction, we obtain a loss
# The loss gradients are back-propagated using loss.backward()

# Because of the privacy engine, norms of the gradients propagated backwards
# are clipped to less than max_grad_norm (ensuring sensitivity)

# Gradients for a batch are averaged, Gaussian noise is added based on the noise_multiplier

# optimizer takes a step in the direction opposite to that of the largest noisy gradient


def train(model, train_loader, optimizer, epoch, device, delta):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # epsilon and delta help us determine shape and size of the Gaussian noise distribution

    # opacus neatly adds support for Differential Privacy to a PyTorch model by attaching a privacy engine, so the training process is the same
    # epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)

    print(
        f"Train Epoch: {epoch} t"
        f"Loss: {np.mean(losses):.6f} "
        # f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
    )


for epoch in range(1, 11):
    train(model, train_loader, optimizer, epoch, device="cpu", delta=1e-5)


# Source for example: https://openmined.org/blog/differentially-private-deep-learning-using-opacus-in-20-lines-of-code/
