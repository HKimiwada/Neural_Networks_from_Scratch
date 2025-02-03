import numpy as np
from keras.datasets import mnist
from mnist_preprocess import preprocess_data

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from MLP.dense import Layer_Dense
from convolutional import Convolutional
from reshape import Reshape
from MLP.activation_function import Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    # Convolutional layer - Make sure to specify input and kernel sizes correctly
    Convolutional((1, 28, 28), 3, 5),  # Example: 1 input channel, 28x28 image, 5 filters (channels)
    Sigmoid(),  # Activation after convolution
    Reshape((5, 26, 26), (5 * 26 * 26,)),  # Flatten the data after convolution (output size depends on kernel)
    Layer_Dense(5 * 26 * 26, 100),  # Dense layer - 100 neurons
    Sigmoid(),  # Activation function for dense layer
    Layer_Dense(100, 2),  # Output layer with 2 neurons (for binary classification)
    Sigmoid()  # Output activation
]

# train
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")

