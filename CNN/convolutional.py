import numpy as np
from scipy import signal

class Convolutional:
    def __init__(self, input_shape, kernel_size, depth):
        # input_shape = (height, width, depth)
        # kernel_size = kernel's height = kernel's width
        # depth = number of kernels
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases) # convolution + bias = output
        for i in range(self.depth):
            # Each kernel is convolved with each input channel
            for j in range(self.input_depth):
                self.output[i] += signal.convolve2d(input[j], self.kernels[i][j], "valid")
        
        return self.output
