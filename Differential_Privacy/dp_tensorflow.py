from abc import ABC, abstractmethod
import os, pytest, math
import numpy as np
import torch
import tensorflow as tf
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy


class tensorflow_fw(DPFramework):
    def __init__(self):
        tf.compat.v1.disable_v2_behavior()
        tf.get_logger().setLevel("DEBUG")
        self.train_data = None
        self.test_data = None
        self.model = None

    def processData():
        # Use the MNIST dataset, just as we did with Opacus
        train, test = tf.keras.datasets.mnist.load_data()
        train_data, train_labels = train
        test_data, test_labels = test

        # add preprocessing operations here

        return train_data, train_labels, test_data, test_labels
