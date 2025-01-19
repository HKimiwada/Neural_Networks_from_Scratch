# Problem statement: the dataset is dots in spiral form. There are several spirals. Each dot in the spiral is one of three colors.
# Create NN to classify each dot. Eseentially, the input will be two dimensional (X,Y) and output is three dimensional.
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Dense Layer Class:
# Class that, when given number of neurons and number inputs, generates output
# Class randomly assigns, weights, and biases to each neuron.
# np.dot(input, Weight) + bias -> Weight (1st column: Weight of neuron 1, 2nd column: Weight of neuron 2 etc...)
# Weights matrix: inputs x neurons, Bias: neuron number
class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
       # Function to initialize weights and biases based on number of inputs and neurons
       self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # Random sample from gaussian distribution, 0.01 to make data small
       self.biases = np.zeros((1,n_neurons))
    
    def forward(self, inputs):
        # Forward pass (Calculating output from input, weights, biases)
        self.outputs = np.dot(inputs, self.weights) + self.biases
        
# Testing Dense Layer Class:
# Create Dataset
X, y = spiral_data(samples=100, classes=3) # generates 100 samples of coordinates in form: [X,Y]
# Create Dense Layer (2 input features: X, Y ) (3 neurons)
dense_layer_test = Layer_Dense(2,3)
# Perform forward pass of training data through layer
dense_layer_test.forward(X)
# Output should be 100 rows x 3 columns (each column represents the output of each neuron, each row represents each data point)
# First few examples:
print(dense_layer_test.outputs[:5]) # 0 through 4
