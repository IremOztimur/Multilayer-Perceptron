from model.layer import Dense
from model.loss import LossCategoricalCrossentropy
from model.metrics import calculate_accuracy
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

    def forward_propagation(self, inputs):
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

    def update_params(self):
        for layer in self.layers:
            layer.weights += -self.learning_rate * layer.dweights
            layer.biases += -self.learning_rate * layer.dbiases

    def train(self, X, y, n_epochs, batch_size):
        n_samples = X.shape[0]
        for epoch in range(n_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_X = X[start:end]
                batch_y = y[start:end]
                
                predictions = self.forward_propagation(batch_X)
                loss = self.loss_function.calculate(predictions, batch_y)
                accuracy = calculate_accuracy(predictions, batch_y)
                
                print(f"step: {start//8}/{n_samples//8} \033[91mLoss: {loss}\033[0m, \033[92mAccuracy: {accuracy}\033[0m")
                
                self.backward_propagation(predictions, batch_y)
                self.update_params()
            
            print("*"*51)
            print(f"Epoch {epoch + 1}/{n_epochs}, \033[91mLoss: {loss}\033[0m, \033[92mAccuracy: {accuracy}\033[0m")
            print("*"*51)

    def predict(self, X):
        predictions = self.forward_propagation(X)
        return np.argmax(predictions, axis=1)
