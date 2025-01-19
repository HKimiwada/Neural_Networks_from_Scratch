# Coding cross entropy loss = - Sum of P(true) * log(P(actual))
# Testing cross entropy loss
import numpy as np

softmax_outputs = np.array([[0.7,0.1,0.2],[0.1,0.5,0.4],[0.02,0.9,0.08]])
class_targets = [0,1,1]

softmax_outputs = softmax_outputs[[0,1,2],class_targets]
neg_log = -np.log(softmax_outputs)
average_loss = np.mean(neg_log)
print(f"Average Cross Entropy Loss: {average_loss}")

# If One hot encoded, begin by dot product class predictions and predictions matrix 
y_true_check = np.array([[1,0,0],[0,1,0],[0,1,0]])
y_predictions = np.array([[0.7,0.2,0.1],[0.1,0.5,0.4],[0.02,0.9,0.08]])
neg_loss_y = -np.log((y_true_check*y_predictions)[[0,1,2],class_targets]) # multiplication does element wise multiplication on array.
average_loss_y = np.mean(neg_loss_y)
print(f"Average Cross Entropy Loss (One-Hot Encoded): {average_loss_y}")
