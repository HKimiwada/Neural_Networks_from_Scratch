# Creating Neural Network from scratch (using numpy for linear algebra)
# main.py creates the neural network.
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from dense import Layer_Dense
from activation_function import ReLU_Activation_Function
from activation_function import Softmax_Activation_Function
from cross_entropy_loss import Categorical_CrossEntropy_Loss

nnfs.init()
X, y = spiral_data(samples=100, classes=3)

## Forward Pass Without Loss
# Defining the layers
layer_1 = Layer_Dense(2,3)
relu_1 = ReLU_Activation_Function()
layer_2 = Layer_Dense(3,3)
softmax_1 = Softmax_Activation_Function()
loss_function = Categorical_CrossEntropy_Loss()

# Creating the Neural Network
layer_1.forward(X)
relu_1.forward(layer_1.outputs)
layer_2.forward(relu_1.outputs)
softmax_1.forward(layer_2.outputs)
loss = loss_function.calculate(softmax_1.outputs,y)
print(f"Loss: {loss}")


    


