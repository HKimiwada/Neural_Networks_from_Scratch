# Coding cross entropy loss = - Sum of P(true) * log(P(actual))
# Testing cross entropy loss
import numpy as np

softmax_outputs = np.array([[0.7,0.1,0.2],[0.1,0.5,0.4],[0.02,0.9,0.08]])
class_targets = [0,1,1]

logits = softmax_outputs[[0,1,2],class_targets]
neg_log = -np.log(logits)
average_loss = np.mean(neg_log)
print(f"Average Cross Entropy Loss: {average_loss}")
