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
# Do clipping before -log because 0 or 1 could mess up loss calculations and gradient updates.
neg_loss_y = -np.log((y_true_check*y_predictions)[[0,1,2],class_targets]) # multiplication does element wise multiplication on array.
average_loss_y = np.mean(neg_loss_y)
print(f"Average Cross Entropy Loss (One-Hot Encoded): {average_loss_y}")

class Loss:
    # Calculates loss given model output and ground truth values
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
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]

        # One-Hot Encoding
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods # Have to get mean of array to optimize loss

if __name__ == "__main__":
    print()
    print("Running Classes to calculate loss")
    true_test = np.array([[1,0,0],[0,1,0],[0,1,0]])
    softmax_outputs_test = np.array([[0.7,0.2,0.1],[0.1,0.5,0.4],[0.02,0.9,0.08]])
    loss_function = Categorical_CrossEntropy_Loss()
    loss_output = loss_function.calculate(softmax_outputs_test, true_test)
    print(f"Losses Calculate using classes: {loss_output}")


