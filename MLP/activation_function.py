import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from MLP.dense import Layer_Dense

# Activation Function: Introduces non-linearity in order to capture any type of function
class ReLU_Activation_Function: 
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0,inputs) # ReLu 

    def backward(self, dvalues):
        # dinputs would stay the same, unless value smaller than 0.
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# ReLu gives out all real-numbers (to transform to prob. use different activation function)
# Softmax Activation Function: Transform output to probability (0,1)
# Subtract largest number in row to make sure softmax value isn't too large.

# Softmax Activation Class
# Softmax returns 33,333% for each color because the weights are set close to zero (0.01*weights)
class Softmax_Activation_Function:
    def forward(self,inputs):
        exp_value = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        prob = exp_value / np.sum(exp_value,axis=1,keepdims=True)
        self.outputs = prob

if __name__ == "__main__":
    # Testing Softmax 
    inputs = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
    exp_value = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
    prob = exp_value / np.sum(exp_value,axis=1,keepdims=True)
    print(f"Probability from input: {prob}")


