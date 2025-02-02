# Coding cross entropy loss = - Sum of P(true) * log(P(actual))
# Testing cross entropy loss
import numpy as np
from MLP.activation_function import Softmax_Activation_Function

class Loss:
    # Calculates loss given model output and ground truth values
    # Loss class extended to other specific loss classes
    def calculate(self,pred,true):
        losses = self.forward(pred,true)
        mean_loss = np.mean(losses)
        return mean_loss

class Categorical_CrossEntropy_Loss(Loss):
    def forward(self,y_pred,y_true):
        # Number of samples in batch
        samples = len(y_pred)
        # Clip data to prevent division by 0 or 1 to prevent gradient being zero
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        
        # 1D target (ex. [0,1,2]: for classification)
        if len(y_true.shape) == 1: # Categorical Cross Entropy Loss
            correct_confidences = y_pred_clipped[range(samples),y_true]

        # One-Hot Encoding
        elif len(y_true.shape) == 2: # Sparse categorical cross entropy loss
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods # Have to get mean of array to optimize loss

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Nuble of labels in each sample
        labels = len(dvalues[0])

        # If labels are sparse, transform to one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Combine Softmax and Cross Entropy Loss for faster backward step
class Softmax_Cross_Entropy_Loss:
    # Create activation and loss function objects
    def __init__(self):
        self.activation = Softmax_Activation_Function()
        self.loss = Categorical_CrossEntropy_Loss()
    
    # Forward Pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs
        
        # Calculate and return loss value
        return self.loss.calculate(self.outputs,y_true)
    
    # Backward Pass
    def backward(self, dvalues,y_true):
        # Number of samples
        samples = len(dvalues)
        # Turn one-hot encodings to discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

if __name__ == "__main__":
    softmax_outputs = np.array([[0.7,0.1,0.2],[0.1,0.5,0.4],[0.02,0.9,0.08]])
    class_targets = [0,1,1]

    softmax_outputs = softmax_outputs[[0,1,2],class_targets]
    neg_log = -np.log(softmax_outputs)
    average_loss = np.mean(neg_log)
    print(f"Average Cross Entropy Loss: {average_loss}")

    # If One hot encoded, begin by dot product class predictions and predictions matrix 
    y_true_check = np.array([[1,0,0],[0,1,0],[0,1,0]])
    y_predictions = np.array([[0.7,0.2,0.1],[0.1,0.5,0.4],[0.02,0.9,0.08]])
    # Do clipping before -log because 0 or 1 could mess up loss calculations and gradient updates.
    neg_loss_y = -np.log((y_true_check*y_predictions)[[0,1,2],class_targets]) # multiplication does element wise multiplication on array.
    average_loss_y = np.mean(neg_loss_y)
    print(f"Average Cross Entropy Loss (One-Hot Encoded): {average_loss_y}")
    
    print()
    print("Running Classes to calculate loss")
    true_test = np.array([[1,0,0],[0,1,0],[0,1,0]])
    softmax_outputs_test = np.array([[0.7,0.2,0.1],[0.1,0.5,0.4],[0.02,0.9,0.08]])
    loss_function = Categorical_CrossEntropy_Loss()
    loss_output = loss_function.calculate(softmax_outputs_test, true_test)
    print(f"Losses Calculate using classes: {loss_output}")


