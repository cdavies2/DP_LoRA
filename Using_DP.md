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