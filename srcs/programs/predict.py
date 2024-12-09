import argparse
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.neural_network import NeuralNetwork
from model.layer import Dense
from model.activation import ReLU, Softmax, Sigmoid
from model.preprocess import to_categorical

def load_model(file_path):
    """Load the trained model from a .npy file."""
    model_data = np.load(file_path, allow_pickle=True).item()
    nn = NeuralNetwork(learning_rate=model_data['learning_rate'])
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

        nn.layers.append(layer)
    print(f"\033[95m> model loaded from {file_path}\033[0m")
    return nn

def save_predictions(predictions, output_path):
    """Save predictions to a CSV file."""
    pd.DataFrame(predictions, columns=['Predicted']).to_csv(output_path, index=False)
    print(f"\033[90m> predictions saved to {output_path}\033[0m")

def load_data(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path)
    X = df.values[:, :-1]
    y = df.values[:, -1]
    y = to_categorical(y)
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Prediction Program")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model (npy format).")
    parser.add_argument("--input", type=str, required=True, help="Path to the input dataset (CSV format).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the predictions (CSV format).")
    args = parser.parse_args()

    nn = load_model(args.model)

    X_test, y_test = load_data(args.input)

    predictions = nn.predict(X_test)

    save_predictions(predictions, args.output)

    y_true = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_true)
    print("*"*29)
    print(f"\033[92m> Accuracy\033[0m on test data: {accuracy:.2f}")
    print("*"*29)

if __name__ == "__main__":
    main()