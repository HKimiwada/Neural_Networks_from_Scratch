# Creating Neural Network from scratch (using numpy for linear algebra)
# main.py creates the neural network.

# Problem statement: the dataset is dots in spiral form. There are several spirals. Each dot in the spiral is one of three colors.
# Create NN to classify each dot. Eseentially, the input will be two dimensional (X,Y) and output is three dimensional.

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from dense import Layer_Dense
from activation_function import ReLU_Activation_Function
from activation_function import Softmax_Activation_Function
from cross_entropy_loss import Categorical_CrossEntropy_Loss
from cross_entropy_loss import Softmax_Cross_Entropy_Loss
from optimizer import Optimizer_GD

nnfs.init()
X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:,0],X[:,1],c=y,cmap="brg")
# plt.show()

# Define Classes
dense1 = Layer_Dense(2,64)
activation1 = ReLU_Activation_Function()
dense2 = Layer_Dense(64,3)
loss_activation = Softmax_Cross_Entropy_Loss()
optimizer = Optimizer_GD(decay=1e-3)

# Training Loop 
for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    loss = loss_activation.forward(dense2.outputs,y)

    predictions = np.argmax(loss_activation.outputs,axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions == y)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward Pass
    loss_activation.backward(loss_activation.outputs,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
    


