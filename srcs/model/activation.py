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
        

class Softmax(Activation):
    def forward(self, inputs):
        # Subtract the max value from each input for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities