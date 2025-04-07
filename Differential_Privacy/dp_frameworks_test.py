import torch
from torchvision import datasets, transforms
import numpy as np
from opacus import PrivacyEngine
from tqdm import tqdm
import pytest
import tensorflow as tf
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import dp_opacus
import dp_tensorflow

# # Training a simple PyTorch classification model


@pytest.mark.parametrize(
    "model", [(opacus_fw, tensorflow_fw)], ids=["Opacus", "Tensorflow"]
)
def test_import(model):
    if model == opacus_fw:
        framework = dp_opacus.opacus_fw
    else:
        framework = dp_tensorflow.tensorflow_fw

    train_data, test_data = framework.processData()

    assert train_data.min() == 0.0
    assert train_data.max() == 1.0
    assert test_data.min() == 0.0
    assert test_data.max() == 1.0


# # Use the built-in MNIST dataset from PyTorch
train_loader, test_loader = dp_opacus.opacus_fw.processData()

# # This Sequential PyTorch neural network model predicts the label of images from the MNIST dataset
# # The network has 2 convolutional layers, and 2 fully connected layers
# # (along with ReLU layers to set any negative values to zero and maxpooling layers to reduce size of features),
# # The first layer accepts images of size (28 x 28) as input,
# # the final returns a vector of probabilities of size (1 x 10),
# #  predicting the digit each image represents

# # A Stochastic Gradient Descent (SGD) optimizer improves the model, and the learning rate is set to 0.05

# model = torch.nn.Sequential(
#     torch.nn.Conv2d(
#         1, 16, 8, 2, padding=3
#     ),  # applies 2D convolution over an input signal composed of multiple input planes.
#     # There is one in channel, 16 out channels, a kernel size of 8, stride of 2, and padding of 3
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(
#         2, 1
#     ),  # applies a 2D max pooling over an input signal composed of several input planes
#     # kernel size is 2, stride is 1
#     torch.nn.Conv2d(16, 32, 4, 2),
#     # 16 input, 32 output, kernel is 4, stride is 2, no padding
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(2, 1),
#     torch.nn.Flatten(),  # flattens input by reshaping it into a 1-dimensional tensor
#     torch.nn.Linear(
#         512, 32
#     ),  # applies an affine linear transformation (y=xA^t + b) to the incoming data.
#     # input has a size of 32 * 4 * 4 (512) and output has a size of 32
#     torch.nn.ReLU(),
#     torch.nn.Linear(
#         32, 10
#     ),  # for this linear transformation, input has a size of 32, output has a size of 10
# )
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
# # applies stochastic gradient descent


# # Attaching a PrivacyEngine makes our model Differentially Private
# # Said engine also helps us track our privacy budget

# # sample_size is the size of the sample set
# # noise_multiplier is the ratio of the standard deviation
# # of Gaussian noise distribution to L2 sensitivity of the function

# # Global sensitivity is the maximum difference in output when one change is made to a dataset
# # L2 is the square root of the sum of the squares of a vector, L1 is the sum of the vector's elements

# # max_grad_norm is the value of the upper bound of the L2 norm of the loss gradients

# privacy_engine = PrivacyEngine(secure_mode=False)


# model, optimizer, train_loader = privacy_engine.make_private(
#     module=model,
#     optimizer=optimizer,
#     data_loader=train_loader,
#     noise_multiplier=1.3,
#     max_grad_norm=1.0,
# )

# # When training a model, we iterate over the training batches of data
# # and make the model predict the data label.
# # Based on the prediction, we obtain a loss
# # The loss gradients are back-propagated using loss.backward()

# # Because of the privacy engine, norms of the gradients propagated backwards
# # are clipped to less than max_grad_norm (ensuring sensitivity)

# # Gradients for a batch are averaged, Gaussian noise is added based on the noise_multiplier

# # optimizer takes a step in the direction opposite to that of the largest noisy gradient


# def train(model, train_loader, optimizer, epoch, device, delta):
#     model.train()
#     criterion = torch.nn.CrossEntropyLoss()
#     losses = []
#     for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())

#     # epsilon and delta help us determine shape and size of the Gaussian noise distribution

#     # opacus neatly adds support for Differential Privacy to a PyTorch model by attaching a privacy engine, so the training process is the same
#     # epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)

#     print(
#         f"Train Epoch: {epoch} t"
#         f"Loss: {np.mean(losses):.6f} "
#         # f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
#     )


# for epoch in range(1, 11):
#     train(model, train_loader, optimizer, epoch, device="cpu", delta=1e-5)


# Source for example: https://openmined.org/blog/differentially-private-deep-learning-using-opacus-in-20-lines-of-code/

# Implement Differential Privacy with Tensorflow

train_data, train_labels, test_data, test_labels = (
    dp_tensorflow.tensorflow_fw.processData()
)

# train_data = np.array(train_data, dtype=np.float32) / 255
# test_data = np.array(test_data, dtype=np.float32) / 255
# # dividing by 255 converts image values to values in a range of 0 to 1, normalizing it

# train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)

# test_data = test_data.reshape(
#     test_data.shape[0], 28, 28, 1
# )  # reshapes image data to a 28x28 pixel image

# train_labels = np.array(train_labels, dtype=np.int32)
# test_labels = np.array(
#     test_labels, dtype=np.int32
# )  # create arrays with the data labels


# train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
# test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)


# this converts the train_labels and test_labels vector into binary classes matrices with 10 classes each
# def test_preprocessing():
#     assert train_data.min() == 0.0
#     assert train_data.max() == 1.0
#     assert test_data.min() == 0.0
#     assert test_data.max() == 1.0


# check that preprocessing was effective, and the training and test data is all in the range between 0 and 1

epochs = 3
batch_size = 250
# set learning model hyperparameters

l2_norm_clip = 1.5  # maximum Euclidean norm of each gradient that is applied to update model parameters, bounds optimizer sensitivity to training points
noise_multiplier = 1.3
# amount of noise sampled/added to gradients, more noise = more private
num_microbatches = 250  # each batch of data is split into microbatches, each with a single training example. Number of microbatches should evenly divide batch size, allowing to include multiple samples and reduce overhead
learning_rate = 0.25  # higher noise, lower learning rate helps convergence

# l2_norm_clip, noise_multiplier, and microbatches are privacy-specific hyperparameters


def test_microbatches():
    assert batch_size % num_microbatches == 0
    # num_microbatches should divide evenly into batch_size


# now build a convolutional neural network learning model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, 8, strides=2, padding="same", activation="relu", input_shape=(28, 28, 1)
        ),
        # above creates a convolution kernel that convolves over one dimension to produce a tensor of outputs
        # there are 16 filters in the convolution, a window/kernel size of 8, 2 strides, even padding on each side, rectified linear unit activation, and a 28x28 input shape
        tf.keras.layers.MaxPool2D(2, 1),
        # downsamples input along its spatial dimensions by tking the maximum value over an input window (in this case, said window length is 2, shifted by 1 stride)
        tf.keras.layers.Conv2D(32, 4, strides=2, padding="valid", activation="relu"),
        # the next kernel had no padding, 32 filters, and a kernel size of 4
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Flatten(),  # flattens input
        tf.keras.layers.Dense(32, activation="relu"),
        # activates the rectified linear union function, output space is 32
        tf.keras.layers.Dense(10),
        # output space is 10
    ]
)

# define the optimizer and loss function
# the loss is a vector of losses per-example
optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate,
)

loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, reduction=tf.losses.Reduction.NONE
)
# the above calculates softmax loss, it measures the difference between predicted probability distribution and true distribution of classes.
# the predicted targets are expected to be a logits tensor, no reduction

# train the model
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model.fit(
    train_data,
    train_labels,
    epochs=epochs,
    validation_data=(test_data, test_labels),
    batch_size=batch_size,
)

# perform privacy analysis, which measures how much an adversary could improve their guess about properties of any individual training point by observing the outcome of the training procedure

# lower privacy budget ensures a stronger privacy guarantee (as it is harder then for a single training point to affect the outcome of learning)

# the two metrics used to express DP guarantee are delta (bounds probability of the guarantee not holding, should be less than the inverse of the size of the training dataset), and epsilon (bounds probability of a model's output varying by including/excluding a single training point).

# compute_dp_sgd_privacy computes epsilon given delta, and its hyperaprameters are the number of points in the training data, batch_size, noise_multiplier, training epochs, and value of delta

# tensorflow_privacy.compute_dp_sgd_privacy_statement(n=train_data.shape[0],
#                        batch_size=batch_size,
#                        noise_multiplier=noise_multiplier,
#                        epochs=epochs,
#                        delta=1e-5)
print(
    tensorflow_privacy.compute_dp_sgd_privacy_statement(
        number_of_examples=train_data.shape[0],
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        num_epochs=epochs,
        delta=1e-5,
    )
)

# Source for example: https://www.tensorflow.org/responsible_ai/privacy/tutorials/classification_privacy
