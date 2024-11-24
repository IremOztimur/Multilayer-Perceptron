import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    @abstractmethod
    def forward(self, inputs):
        pass
        

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_inputs, n_neurons)
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases