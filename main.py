# Creating Neural Network from scratch (with only numpy for linear algebra)
# main.py is for testing the functionality of numpy and python
import numpy as np

# np.sum(A) = np.sum(A, axis=None) -> Adds all elememts and returns scalar
# np.sum(A, axis=0) -> Adds together same elements each row: A_11 + B_21 + C_31, A_12 + B_22 + C_32
# np.sum(A, axis=1) -> Adds together same elements of column 

# Broadcasting allows for adding biases: [1,2,3] -> [[1,2,3],[1,2,3],[1,2,3]] -> Extends by maintaing shape
A = [[1,2,3],[4,5,6],[7,8,9]]
# [1,2,3]
# [4,5,6]
# [7,8,9]
print(np.sum(A)) #45

print(np.sum(A,axis=0)) # [12,15,18]
print(np.sum(A,axis=0,keepdims=True)) # [[12,15,18]]

print(np.sum(A,axis=1)) # [6,15,24]
print(np.sum(A,axis=1,keepdims=True)) # [[6],[15],[24]]

print(A-np.max(A,axis=0,keepdims=True))
# np.max: [[7,8,9]]
# [-6,-6,-6]
# [-3,-3,-3]
# [0,0,0]

