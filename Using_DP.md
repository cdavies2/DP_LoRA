# The DP-SGD Algorithm
* Any change to the training process (permuting data, running another task on the same GPU) impacts the resulting model significantly. This means you can't just measure how different a model's weights are in two scenarios to determine how much your data impacts a model
* With differential privacy, we examine the probabilites of observing weights in different models, and change is guaranteed, but not by more than a predefined amount.
* A process is ε-differentially private if for every possible output x, the probability that said output is observed never differs by more than exp(ε) between the two scenarios (with or without your data)
* The higher ε is, the less private you are, so ε is often referred to as the "privacy/loss budget"
* If we can prove that all weights of two models are observed with probabilities that lie within a predefined boundary of exp(ε) of each other, we can claim the training procedure is differentially private.
* The Fundamental Law of Information Recovery states answers to queries cannot be overly precise, and error must grow with the number of answers if we want to avoid near total reconstruction of a dataset.
* "Simple composition" means that if you take a measurement from a mechanism with privacy budget ε1, and another from a mechanism with budget ε2, the total privacy budget is ε1 + ε2. A single query has ε=1 and three queries have a budget of ε=3
* Differential privacy is preserved by post-processing, meaning the results of running arbitrary computations on top of differentially private output won't roll back the ε.
* A differentially private solution involves these three steps....
    1. Our mechanism will be randomized (it will use noise)
    2. Our final privacy claim depends on the total number of interactions with the data
    3. We can post-process results of a differentially private computation any way we want (as long as we don't peek into the private dataset again)

## Application to Machine Learning
* In most applications of ML, inputs come without explicit user identifiers (Federated Learning is the exception), so we'll default to protecting privacy of a specific sample.
* Private prediction considers privacy of model outputs only, but this prevents us from releasing the model. As such, it is preferable to insert the DP mechanism during model training.

## DP-SGD
* DP-SGD (Differentially-Private Stochastic Gradient Descent) modifies the minibatch stochastic optimization process to make it differentially private. 
* Training a model in PyTorch can be done through access to its parameter gradients (gradients of the loss with respect to each parameter)
* Because the PyTorch optimizer is already made to look at parameter gradients, noise can be added directly into it. The code below shows how it is done
```
optimizer = torch.optim.SGD(lr=args.lr)

for batch in Dataloader(train_dataset, batch_size=32):
    x, y = batch
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    
    # Now these are filled:
    gradients = (p.grad for p in model.parameters())
  
    for p in model.parameters():

        # Add our differential privacy magic here
        p.grad += torch.normal(mean=0, std=args.sigma)
        
        # This is what optimizer.step() does
        p = p - args.lr * p.grad
        p.grad.zero_()
```
*  If you don't add enough noise, privacy isn't preserved, but if you add too much, the model is useless.
* The right amount of privacy depends on the largest norm of the gradient in a minibatch, as that is the sample that is at most risk of exposure.
* Using the Gaussian mechanism you can take in two parameters (noise multiplier and bound on the gradient norm).
* If gradients aren't bounded, bound them yourself by a clipping threshold. This prevents the model from learning more than a set quantity from a given training sample, no matter how different it is from the rest.
* We typically process data in batches, but we want the batch for each tensor, not the average of the whole batch. Computing per-sample gradients is slow, as it forces us to run backward steps for one sample at a time. This is called the microbatch method, and it offers simplicity and universal compatibiltiy, at a low trainings speed. 
* Opacus uses a faster training method but also performs some extra engineering work.
* The code below follows four steps...
    1. Computes per-sample gradients
    2. Clips them to a fixed maximum norm
    3. Aggregates them back to a single parameter gradient
    4. Adds noise to it
```
from torch.nn.utils import clip_grad_norm_

optimizer = torch.optim.SGD(lr=args.lr)

for batch in Dataloader(train_dataset, batch_size=32):
    for param in model.parameters():
        param.accumulated_grads = []
    
    # Run the microbatches
    for sample in batch:
        x, y = sample
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
    
        # Clip each parameter's per-sample gradient
        for param in model.parameters():
            per_sample_grad = p.grad.detach().clone()
            clip_grad_norm_(per_sample_grad, max_norm=args.max_grad_norm)  # in-place
            param.accumulated_grads.append(per_sample_grad)  
        
    # Aggregate back
    for param in model.parameters():
        param.grad = torch.stack(param.accumulated_grads, dim=0)

    # Now we are ready to update and add noise!
    for param in model.parameters():
        param = param - args.lr * param.grad
        param += torch.normal(mean=0, std=args.noise_multiplier * args.max_grad_norm)
        
        param.grad = 0  # Reset for next iteration
```

* Source: https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3

# Efficient Per-Sample Gradient Computation in Opacus
* Differential privacy focuses on worst-case guarantees, so we must check the gradient of every sample in a batch of data.
* Microbatching yields correct gradients but is very inefficient.

## Vectorized Computation
* Vectorized computation allows Opacus to compute per-sample gradients much faster than microbatching.
* To perform this, we derive the per-sample gradient formula, and implement a vectorized version of it.
* The focus is on simple linear layers (building blocks for multi-layer perceptions)
## Efficient Per-Sample Gradient Computation for MLP
* This focuses on one linear layer in a neural network, with weight matrix W, bias is omitted, and forward pass is denoted by Y=WX where X is input and Y is output.
* With a single sample, X is a vector, with  a batch (like in Opacus) X is a matrix of size dxB, with B columns (B is batch size) where each column is an input vector of dimension d. 
* Output matrix Y is of size rxB where each column is the output vector corresponding to an element in the batch and r is the output dimension.
* In Opacus we need the per-sample derivative of the loss with respect to weights W. First, find the derivative of the loss with respect to weights and later deal with per-sample.
* General form of the chain rule is...
    ∂L/∂z = (∂L/∂y) * (∂y/∂z)
* The simplified version of this equation corresponds to a matrix multiplication in PyTorch. 
* The gradient of loss with respect to the weight relies on the gradient of loss with respect to output Y. Because Opacus requires computation of per-sample gradients, we need the following...
    (∂Lbatch/∂Wi,j) = (∂L/∂Yi'^(b))*Xj^(b)
* The notation Y=WX was used for forward pass of a single layer of a neural network. When the network has more layers, a better notation would be Z^(n+1) = W^(n+1) * Z^(n), where n corresponds to each layer of the neural network. In that case, gradients with respect to any activations Z^(n) are "highway gradients" and gradients with respect to weights are "exit gradients"
* Highway gradients retain per-sample information, but exit gradients do not (highway are per-sample, but exit aren't necessarily)
* To compute sample exit gradients efficiently, we....
    1. Store activations elsewhere
    2. Find a way to access highway gradients.
* PyTorch can do the above with module and tensor hooks. The main ones are...
    1. _Parameter hook_: attaches to a `nn.Module's` Parameter tensor and will always run dduring the backward pass. The signature is `hook(grad) -> Tensor on None`
    2. nn.Module hook, there are two types
        a. _Forward hook_: `hook(module, input, output) -> None or modified output`
        b. _Backward hook_: `hook(module, grad_input, grad_output) -> tuple(Tensor) or None`
* `grad_input` and `grad_output` are tuples that contain the gradients with respect to the inputs and outputs respectively.
* We use two hooks, one forward and one backward. In the forward below, store the activations
```
def forward_hook(module, input, output):
    module.activations = input
```
* In the backward, use `grad_output` (highway gradient) along with stored activations (input to layer) to compute the per-sample gradient
```
def backward_hook(module, grad_input, grad_output):
    module.grad_sample = compute_grad_sample(module.activations, grad_output)
```
* The average gradient of loss with respect to the weights is the result of a matrix multiplication. To get the per-sample gradient, remove the sum reduction, replacing the matrix multiplication with a batched outer product. Torch `einsum` lets us do that in vectorized form, and the method `compute_grad_sample` is defined based on `einsum` throughout our code. This is what said code looks like 
```

import torch
import torch.nn as nn

from .utils import create_or_extend_grad_sample, register_grad_sampler


@register_grad_sampler(nn.Linear)
def compute_linear_grad_sample(
    layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
    """
    Computes per sample gradients for ``nn.Linear`` layer

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    gs = torch.einsum("n...i,n...j->nij", B, A)
    create_or_extend_grad_sample(layer.weight, gs, batch_dim)
    if layer.bias is not None:

        create_or_extend_grad_sample(
            layer.bias,
            torch.einsum("n...k->nk", B),
            batch_dim,
        )
```

* Source: https://medium.com/pytorch/differential-privacy-series-part-2-efficient-per-sample-gradient-computation-in-opacus-5bf4031d9e22

# Efficient Per-Sample Gradient Computation for More Layers in Opacus
* A major feature of Opacus is "vectorized computation", meaning it can compute per-sample gradients much faster than microbatching by deriving the per-sample gradient formula and implementing a vectorized version of it. Said per-sample gradient formula is....
    (∂Lbatch/∂Wi,j) = (∂L/∂Yi'^(b))*Xj^(b)
* Gradients with respect to activations, which retain per-sample information, are "highway gradients", while gradients with respect to weights are "exit gradients"
* einsum facilitates vectorized computation

## Extending the Idea to Other Modules
* A linear layer performs a matrix multiplication (`matmul`) between inputs and parameters. All other layers do this, but while carrying additional constraints, like weight sharing in a convolution, or sequential accumulation in the backward pass in an LSTM. 

## Convolution
* A `Conv2D` with a 2x2 kernel operating on an input with just one channel involves more than just a simple matrix multiplication. However, a matrix multiplication along with some reshaping can achieve the same results.
* To implement efficient matrix multiplication using einsum, Opacus performs `unfold`, `matmul`, `reshape`, as seen in the link below
https://github.com/pytorch/opacus/blob/main/opacus/grad_sample/conv.py

## Recurrent: RNN, GRU, and LSTM

* Source: https://pytorch.medium.com/differential-privacy-series-part-3-efficient-per-sample-gradient-computation-for-more-layers-in-39bd25df237
