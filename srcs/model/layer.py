import numpy as np
from abc import ABC, abstractmethod
from model.activation import Softmax


class Layer(ABC):
    def __init__(self, n_inputs, n_neurons, activation, initializer='He'):
        if initializer == 'He':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        elif initializer == 'Xavier':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1. / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation_function = activation
        self.initializer = initializer
    
    @abstractmethod
    def forward(self, inputs):
        pass
    
    @abstractmethod
    def backward(self, gradients):
        pass
        

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons, activation, initializer):
        super().__init__(n_inputs, n_neurons, activation, initializer)
        
    def forward(self, inputs):
        self.inputs = inputs
        self.linear_output = np.dot(inputs, self.weights) + self.biases
        self.activation_function.forward(self.linear_output)
        self.output = self.activation_function.output

    def backward(self, gradients, y_true=None):
        if isinstance(self.activation_function, Softmax) and y_true is not None:
            self.activation_function.backward(gradients, y_true)
        else:
            self.activation_function.backward(gradients)
        self.dweights = np.dot(self.inputs.T, self.activation_function.dinputs)
        self.dbiases = np.sum(self.activation_function.dinputs, axis=0, keepdims=True)
        self.dinputs = np.dot(self.activation_function.dinputs, self.weights.T)