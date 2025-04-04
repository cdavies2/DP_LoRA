from abc import ABC, abstractmethod
import os, pytest, math
import numpy as np
import torch
import tensorflow as tf
import tensorflow_privacy
from dp_opacus import DPFramework
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
                
        train_data = np.array(train_data, dtype=np.float32) / 255
        test_data = np.array(test_data, dtype=np.float32) / 255
        # dividing by 255 converts image values to values in a range of 0 to 1, normalizing it
        
        train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
        
        test_data = test_data.reshape(
            test_data.shape[0], 28, 28, 1
        )  # reshapes image data to a 28x28 pixel image
        
        train_labels = np.array(train_labels, dtype=np.int32)
        test_labels = np.array(
            test_labels, dtype=np.int32
        )  # create arrays with the data labels
        
        
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
        test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

        # this converts the train_labels and test_labels vector into binary classes matrices with 10 classes each

        return train_data, train_labels, test_data, test_labels
