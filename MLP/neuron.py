# Practice code -> Creating simple neurons (not part of NN)
import numpy as np

# Simple Neuron 
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2
outputs = (inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2]) + bias
print(f'Output Simple: {outputs}')

# Layer of Neurons
inputs_layer = [1, 2, 3, 2.5]
weights_layer = [[0.2, 0.8, -0.5, 1.0],
                 [0.5, -0.91, 0.26, -0.5],
                 [-0.26, -0.27, 0.17, 0.87]]
biases_layer = [2, 3, 0.5] 

# Using numpy
np_outputs_neuron = np.dot(inputs, weights) + bias
print(f'Numpy Neuron Output: {np_outputs_neuron}')

np_outputs_layer = np.dot(weights_layer, inputs_layer) + biases_layer
print(f"Numpy Layer Output: {np_outputs_layer}")

# Batching (for when there are multiple rounds of inputs, think in MNIST different digits being entered at the same time)
inputs_batch = [[1, 2, 3, 2.5],[2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
np_outputs_batch = np.dot(inputs_batch, np.transpose(weights_layer)) + biases_layer
print(f'Numpy Batch Output: {np_outputs_batch}')

# Stacking Neuron Layer (Input, Hidden, etc...)
# 4 input neurons and two hidden layers with 3 neurons each = Output
# Output_n = np.dot(x_n-1, W_1^T)  + bias_1 
np_inputs_layered = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
hidden_1_weights = [[0.2, 0.8, -0.5, 1.0],
                 [0.5, -0.91, 0.26, -0.5],
                 [-0.26, -0.27, 0.17, 0.87]]
biases_1 = [2, 3, 0.5] 
hidden_2_weights = [[0.1,-0.14,0.5],[-0.5,0.12,-0.33],[-0.44,0.73,-0.13]]
biases_2 = [-1,2,-0.5]

layer_1_outputs = np.dot(np_inputs_layered, np.transpose(hidden_1_weights)) + biases_1
layer_2_outputs = np.dot(layer_1_outputs, np.transpose(hidden_2_weights)) + biases_2
print(f"Stacked Output using numpy: {layer_2_outputs}")