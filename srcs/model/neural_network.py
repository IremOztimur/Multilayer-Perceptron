from model.layer import Dense
from model.loss import LossCategoricalCrossEntropy
from model.metrics import calculate_accuracy
from model.activation import ReLU, Sigmoid, Softmax 
import numpy as np

class NeuralNetwork:
    layers: Dense
    
    def __init__(self, loss_function=LossCategoricalCrossEntropy()):
        self.layers = []
        self.loss_function = loss_function
        self.optimizer = None
        
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
        self.optimizer.pre_update_params()
        for layer in self.layers:
            self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

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
    
    def save_model(self, file_path):
        model_data = {
            "topology": [], 
            "parameters": []
        }

        for layer in self.layers:
            model_data["topology"].append({
                "n_inputs": layer.n_inputs,
                "n_neurons": layer.n_neurons,
                "activation": type(layer.activation_function).__name__,
                "initializer": layer.initializer
            })

            model_data["parameters"].append({
                "weights": layer.weights,
                "biases": layer.biases
            })
        model_data["learning_rate"] = self.optimizer.learning_rate

        np.save(file_path, model_data, allow_pickle=True)
        print(f"\033[94m> model saved to {file_path}\033[0m")

    
    def load_model(self, file_path):
        model_data = np.load(file_path, allow_pickle=True).item()

        self.layers = []

        for layer_data, param_data in zip(model_data["topology"], model_data["parameters"]):
            activation_function = globals()[layer_data["activation"]]()
            layer = Dense(
                n_inputs=layer_data["n_inputs"],
                n_neurons=layer_data["n_neurons"],
                activation=activation_function,
                initializer=layer_data["initializer"]
            )

            layer.weights = param_data["weights"]
            layer.biases = param_data["biases"]

            self.layers.append(layer)
        print(f"\033[95m> model loaded from {file_path}\033[0m")
        return self
