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