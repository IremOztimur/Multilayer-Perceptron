import numpy as np
from activation import ReLU, Softmax
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, n_inputs, n_neurons, activation):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation_function = activation
        self.activation_derivative = None
    
    @abstractmethod
    def forward(self, inputs):
        pass
        

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons, activation):
        super().__init__(n_inputs, n_neurons, activation)
        
    def forward(self, inputs):
        self.inputs = inputs
        self.linear_output = np.dot(inputs, self.weights) + self.biases
        self.activation_function.forward(self.linear_output)
        self.output = self.activation_function.output
