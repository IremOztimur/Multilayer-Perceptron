from neural_network import NeuralNetwork
from layer import Dense
from activation import ReLU, Softmax
import numpy as np

if __name__ == '__main__':
    xs = np.array([[1, 2, 3], [2, 3, 4]])
    ys = np.array([[0, 1], [1, 0]])
    nn = NeuralNetwork(learning_rate=0.01)
    nn.add(Dense(3, 4, ReLU()))
    nn.add(Dense(4, 3, ReLU()))
    nn.add(Dense(3, 2, Softmax()))
    print(nn.forward_propagation(xs))
    print("Loss: {:.6f}".format(nn.loss_function.calculate(nn.forward_propagation(xs), ys)))