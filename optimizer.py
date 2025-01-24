# File for Optimizers (Gradient Descent, Adam etc...)
# Simple Gradien Descent Optimizer (w/learning rate decay)
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