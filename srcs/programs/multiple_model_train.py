import argparse
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.neural_network import NeuralNetwork
from model.layer import Dense
from model.activation import ReLU, Softmax, Sigmoid
from model.loss import LossBinaryCrossEntropy, LossCategoricalCrossEntropy
from visualizer import plot_metrics, plot_learning_curves
from model.preprocess import to_categorical

def load_data(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path)
    X = df.values[:, :-1]
    y = df.values[:, -1]
    y = to_categorical(y)
    return X, y

def train_and_save_model(X_train, y_train, X_valid, y_valid, model_name, params):
    """
    Train a neural network model and save metrics.

    Args:
        X_train, y_train: Training data.
        X_valid, y_valid: Validation data.
        model_name: Name of the model for identification.
        params: Dictionary of hyperparameters (learning rate, batch size, epochs).
    
    Returns:
        A dictionary with training and validation metrics.
    """
    nn = NeuralNetwork(learning_rate=params['learning_rate'], loss_function=LossCategoricalCrossEntropy())
    nn.add(Dense(n_inputs=X_train.shape[1], n_neurons=16, activation=ReLU(), initializer='He'))
    nn.add(Dense(n_inputs=16, n_neurons=8, activation=Sigmoid(), initializer='Xavier'))
    nn.add(Dense(n_inputs=8, n_neurons=2, activation=Softmax(), initializer='Xavier'))
    
    history = nn.train(
        X_train,
        y_train,
        n_epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=(X_valid, y_valid),
        patience=3
    )

    model_path = f"depo/{model_name}_saved_model.npy"
    nn.save_model(model_path)
    print(f"Model {model_name} saved to {model_path}")
    return history

def main():
    parser = argparse.ArgumentParser(description="Training Program with Multiple Model Comparison")
    parser.add_argument("--train", type=str, required=True, help="Path to the training dataset (CSV format).")
    parser.add_argument("--valid", type=str, required=True, help="Path to the validation dataset (CSV format).")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for training.")
    args = parser.parse_args()

    X_train, y_train = load_data(args.train)
    X_valid, y_valid = load_data(args.valid)


    models = {
        "Model_A": {"learning_rate": args.learning_rate, "batch_size": args.batch_size, "epochs": args.epochs},
        "Model_B": {"learning_rate": args.learning_rate * 0.5, "batch_size": args.batch_size, "epochs": args.epochs},
        "Model_C": {"learning_rate": args.learning_rate * 2, "batch_size": args.batch_size, "epochs": args.epochs},
    }


    all_models_history = {}

    for model_name, params in models.items():
        print(f"\nTraining {model_name}...")
        history = train_and_save_model(X_train, y_train, X_valid, y_valid, model_name, params)
        all_models_history[model_name] = {
            "train": [history["train_loss"], history['train_accuracy']],
            "validation": [history["val_loss"], history['val_accuracy']]
        }

    print("\nPlotting learning curves...")
    plot_learning_curves(all_models_history, metrics=["loss", "accuracy"])

if __name__ == "__main__":
    main()
