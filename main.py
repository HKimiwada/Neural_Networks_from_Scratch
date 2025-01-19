# Creating Neural Network from scratch (with only numpy for linear algebra)
# Create Mock NN (Inputs+Neurons+Relu+Neurons+Softmax)
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from dense import Layer_Dense
from activation_function import ReLu_Activation_Function
from activation_function import Softmax_Activation_Function

nnfs.init()
X, y = spiral_data(samples=100, classes=3)

## Forward Pass Without Loss
# Defining the layers
layer_1 = Layer_Dense(2,3)
relu_1 = ReLu_Activation_Function()
layer_2 = Layer_Dense(3,3)
softmax_1 = Softmax_Activation_Function()

# Creating the Neural Network
layer_1.forward(X)
relu_1.forward(layer_1.outputs)
layer_2.forward(relu_1.outputs)
softmax_1.forward(layer_2.outputs)
print(f"NN Output: {softmax_1.outputs[:5]}")

# Softmax returns 33,333% for each color because the weights are set close to zero (0.01*weights)
    


