from abc import ABC, abstractmethod
import os, pytest, math
import numpy as np
import torch
from torchvision import datasets, transforms
from opacus import PrivacyEngine
from tqdm import tqdm


class DPFramework(ABC):
    @abstractmethod
    def processData(self):
        pass

    def buildModel(self):
        pass

    def makePrivate(self):
        pass

    def trainModel(self):
        pass


class opacus_fw(DPFramework):
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.model = None

    def processData():
        # this may be modified later to import different data
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

        return train_loader, test_loader
