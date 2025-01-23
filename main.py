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

nnfs.init()
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap="brg")
plt.show()


    


