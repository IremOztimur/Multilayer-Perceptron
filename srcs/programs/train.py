import argparse
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.neural_network import NeuralNetwork
from model.layer import Dense
from model.activation import ReLU, Softmax, Sigmoid
from model.loss import LossBinaryCrossEntropy, LossCategoricalCrossEntropy
from visualizer import plot_metrics
from model.preprocess import to_categorical

def load_data(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path)
    X = df.values[:, :-1]
    y = df.values[:, -1]
    y = to_categorical(y)
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Training Program")
    parser.add_argument("--train", type=str, required=True, help="Path to the training dataset (CSV format).")
    parser.add_argument("--valid", type=str, required=True, help="Path to the validation dataset (CSV format).")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for training.")
    args = parser.parse_args()

    X_train, y_train = load_data(args.train)
    X_valid, y_valid = load_data(args.valid)


    nn = NeuralNetwork(learning_rate=args.learning_rate, loss_function=LossCategoricalCrossEntropy())
    nn.add(Dense(n_inputs=X_train.shape[1], n_neurons=16, activation=ReLU(), initializer='He'))
    nn.add(Dense(n_inputs=16, n_neurons=8, activation=Sigmoid(), initializer='Xavier'))
    nn.add(Dense(n_inputs=8, n_neurons=2, activation=Softmax(), initializer='Xavier'))
    
    history = nn.train(
        X_train,
        y_train,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_valid, y_valid),
        patience=3
    )

    model_path = "depo/saved_model.npy"
    nn.save_model(model_path)

    plot_metrics(history)

if __name__ == "__main__":
    main()
