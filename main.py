# Creating Neural Network from scratch (using numpy for linear algebra)
# main.py creates the neural network.
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from dense import Layer_Dense
from activation_function import ReLU_Activation_Function
from activation_function import Softmax_Activation_Function
from cross_entropy_loss import Categorical_CrossEntropy_Loss
from cross_entropy_loss import Softmax_Cross_Entropy_Loss

nnfs.init()
X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:,0],X[:,1],c=y,cmap="brg")
# plt.show()

dense1 = Layer_Dense(2,3)
activation1 = ReLU_Activation_Function()
dense2 = Layer_Dense(3,3)
loss_activation = Softmax_Cross_Entropy_Loss()
dense1.forward(X)
activation1.forward(dense1.outputs)
dense2.forward(activation1.outputs)
loss = loss_activation.forward(dense2.outputs,y)

predictions = np.argmax(loss_activation.outputs,axis=1)
if len(y.shape) == 2:
    y = np.argmax(y,axis=1)
accuracy = np.mean(predictions == y)
print("Accuracy: ", accuracy)

# Backward Pass
loss_activation.backward(loss_activation.outputs,y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Print Gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
    


