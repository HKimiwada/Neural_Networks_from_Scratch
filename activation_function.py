import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from dense import Layer_Dense

# Activation Function: Introduces non-linearity in order to capture any type of function
class ReLu_Activation_Function: 
    def forward(self, inputs):
        self.outputs = np.maximum(0,inputs) # ReLu 

# ReLu gives out all real-numbers (to transform to prob. use different activation function)
# Softmax Activation Function: Transform output to probability (0,1)
# Subtract largest number in row to make sure softmax value isn't too large.

# Testing Softmax 
input = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
exp_value = np.exp(input - np.max(input,axis=1,keepdims=True))
prob = exp_value / np.sum(exp_value,axis=1,keepdims=True)
print(f"Probability from input: {prob}")

# Softmax Activation Class
class Softmax_Activation_Function:
    def forward(self,inputs):
        exp_value = np.exp(input - np.max(input,axis=1,keepdims=True))
        prob = exp_value / np.sum(exp_value,axis=1,keepdims=True)
        self.outputs = prob

# Create Mock NN (Inputs+Neurons+Relu+Neurons+Softmax)
X, y = spiral_data(samples=100, classes=3)

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
