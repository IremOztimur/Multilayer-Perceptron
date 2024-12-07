import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass


class ReLU(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self
    
    def backward(self, gradients):
        self.dinputs = gradients.copy()
        self.dinputs[self.inputs <= 0] = 0
        print(self.dinputs)

class Softmax(Activation):
    def forward(self, inputs):
        # Subtract the max value from each input for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self
    
    def backward(self, gradients, y_true):
        samples = len(gradients)

        self.dinputs = gradients - y_true
        self.dinputs = self.dinputs / samples