from model.layer import Dense
from model.loss import LossCategoricalCrossEntropy
from model.metrics import calculate_accuracy
import numpy as np

class NeuralNetwork:
    layers: Dense
    learning_rate: float
    
    def __init__(self, learning_rate=0.01, loss_function=LossCategoricalCrossEntropy()):
        self.layers = []
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        
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

    def train(self, X, y, n_epochs, batch_size, validation_data=None, patience=3):
        n_samples = X.shape[0]
        best_loss = float('inf')
        patience_counter = 0

        history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [], 'step_loss': []}

        X_val, y_val = validation_data if validation_data else (None, None)

        for epoch in range(n_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            epoch_loss = 0
            epoch_accuracy = 0
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_X = X[start:end]
                batch_y = y[start:end]

                # Forward pass
                predictions = self.forward_propagation(batch_X)
                loss = self.loss_function.calculate(predictions, batch_y)
                accuracy = calculate_accuracy(predictions, batch_y)

                # Accumulate metrics
                epoch_loss += loss
                epoch_accuracy += accuracy
                history['step_loss'].append(loss)

                # Backward pass and parameter update
                self.backward_propagation(predictions, batch_y)
                self.update_params()

            # Average metrics over all batches
            epoch_loss /= n_samples // batch_size
            epoch_accuracy /= n_samples // batch_size
            history['train_loss'].append(epoch_loss)
            history['train_accuracy'].append(epoch_accuracy)

            # Validation metrics
            if X_val is not None and y_val is not None:
                val_predictions = self.forward_propagation(X_val)
                val_loss = self.loss_function.calculate(val_predictions, y_val)
                val_accuracy = calculate_accuracy(val_predictions, y_val)

                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                print(f"Epoch {epoch + 1}/{n_epochs}, "
                    f"\033[91mTrain Loss: {epoch_loss:.4f}\033[0m, \033[92mTrain Accuracy: {epoch_accuracy:.4f}\033[0m, "
                    f"\033[91mVal Loss: {val_loss:.4f}\033[0m, \033[92mVal Accuracy: {val_accuracy:.4f}\033[0m")

                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            else:
                print(f"Epoch {epoch + 1}/{n_epochs}, \033[91mLoss: {epoch_loss:.4f}\033[0m, \033[92mAccuracy: {epoch_accuracy:.4f}\033[0m")

        return history

    def predict(self, X):
        predictions = self.forward_propagation(X)
        return np.argmax(predictions, axis=1)
