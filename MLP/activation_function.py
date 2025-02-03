import numpy as np
import nnfs
from nnfs.datasets import spiral_data

class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

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

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

    def forward(self, input_data):
        # Ensure input_data is used correctly here instead of self.input
        self.input = input_data
        return self.activation(self.input)

    def backward(self, dvalues):
        self.dvalues = dvalues * (self.output * (1 - self.output))  # Sigmoid derivative
        return self.dvalues

if __name__ == "__main__":
    # Testing Softmax 
    inputs = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
    exp_value = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
    prob = exp_value / np.sum(exp_value,axis=1,keepdims=True)
    print(f"Probability from input: {prob}")


