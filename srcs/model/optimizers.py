import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """Base class for optimizers."""
    @abstractmethod
    def pre_update_params(self):
        pass
    @abstractmethod
    def update_params(self, layer):
        pass
    @abstractmethod
    def post_update_params(self):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.0, decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            layer.weight_momentums = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.bias_momentums = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases

            layer.weights += layer.weight_momentums
            layer.biases += layer.bias_momentums
        else:
            # Vanilla SGD
            layer.weights += -self.learning_rate * layer.dweights
            layer.biases += -self.learning_rate * layer.dbiases
            
    def post_update_params(self):
        self.iterations += 1


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            # Initialize momentums and caches
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentums
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Correct bias
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update caches
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * (layer.dweights ** 2)
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * (layer.dbiases ** 2)

        # Correct bias for cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Apply updates
        layer.weights -= self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7, decay=0.0):
        """
        RMSProp optimizer.
        Args:
            learning_rate: Learning rate for updates.
            rho: Decay rate for the moving average of squared gradients.
            epsilon: Small value to avoid division by zero.
            decay: Learning rate decay over time.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * (layer.dweights ** 2)
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * (layer.dbiases ** 2)

        layer.weights -= self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases -= self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
