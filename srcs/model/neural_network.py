from layer import Dense
from loss import LossCategoricalCrossentropy
import numpy as np

class NeuralNetwork:
    layers: Dense
    learning_rate: float
    
    def __init__(self, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate
        self.loss_function = LossCategoricalCrossentropy() # BinaryCrossentropy() will be replaced
        
    def add(self, layer):
        if not isinstance(layer, Dense):
            raise TypeError("The added layer must be an instance of "
                            f"class Dense. Found: {layer}")
        self.layers.append(layer)

    def forward_propagation(self,inputs):
        feed = inputs
        for layer in self.layers:
            layer.forward(feed)
            feed = layer.output
        return feed

    def backward_propagation(self, output, y):
        self.loss_function.backward(output, y)
        gradients = self.loss_function.dinputs
        
        for layer in reversed(self.layers):
            layer.backward(gradients)
            gradients = layer.dinputs


    def train(self, X, y, n_epochs, batch_size):
        pass