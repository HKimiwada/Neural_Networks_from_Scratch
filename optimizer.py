# File for Optimizers (Gradient Descent, SGD, Adagrad, RMSprop, Adam)
import numpy as np

# Gradient Descent Optimizer (w/learning rate decay) 
class Optimizer_GD:
    def __init__(self, learning_rate=1, decay=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
    
    # Call before any parameter updates (to do learning rate decay)
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

    def post_update_params(self):
        self.iterations += 1

# Optimizer with Momentum (Simple Gradient Descent + Momentum)
class Optimizer_SGD:
    def __init__(self, learning_rate=1, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    # Call before any parameter updates (to do learning rate decay)
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer):
        if self.momentum:
            # if no momentum arrays, create momentum array with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            # Weight updates with momentum. Takes previous weight updates multiplied by some hyperparameter factor
            # and adds to current gradient update
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates # Update momentum array

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates # Update momentum array

        else: # No momentum so Simple Gradient Descent
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        # update weights and biases with either updates from momentum or simple gradient descent
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

# Adagrad Optimizer (optimizes learning rate for each parameter)
class Optimizer_Adagrad:
    def __init__(self, learning_rate=1, decay=0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call before parameter updates (for learning rate decay) 
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)
    
    # Update Parameters
    def update_params(self, layer):
        # If layer has no cache array (of past graduebts), create one filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Update weights and biases with adaptive learning rate for each parameter
        # (learning rate divided by square root of cache + epsilon) -> epsilon is hyperparameter to prevent division by zero
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

# RMSprop Optimizer (optimizes learning rate, but with decay to prevent learning rate from getting too small)
class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    # Call before any parameter updates (to do learning rate decay)
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)
    
    # Update Parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update cache with squared + decayed gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD update with adaptive learning rate + normalization
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        self.iterations += 1