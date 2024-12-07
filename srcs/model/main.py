from neural_network import NeuralNetwork
from layer import Dense
from activation import ReLU, Softmax
import numpy as np
from metrics import calculate_accuracy

if __name__ == '__main__':
    xs = np.array([[1, 2, 3], [2, 3, 4]])
    ys = np.array([[0, 1], [1, 0]])
    nn = NeuralNetwork(learning_rate=0.01)
    nn.add(Dense(3, 4, ReLU()))
    nn.add(Dense(4, 3, ReLU()))
    nn.add(Dense(3, 2, Softmax()))
    forward_feed = nn.forward_propagation(xs)
    print(forward_feed)
    loss = nn.loss_function.forward(forward_feed, ys)
    
    print("Loss: {:.5f}".format(nn.loss_function.calculate(forward_feed, ys)))
    print("Accuracy: {:.5f}".format(calculate_accuracy(forward_feed, ys)))
    
    nn.backward_propagation(forward_feed, ys)