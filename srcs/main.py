from model.neural_network import NeuralNetwork
from model.layer import Dense
from model.activation import ReLU, Softmax, Sigmoid
from model.loss import LossCategoricalCrossEntropy, LossBinaryCrossEntropy
import numpy as np
from model.preprocess import preprocess_data_from_path, train_test_split, to_categorical
from visualizer import plot_metrics, plot_step_loss
import matplotlib.pyplot as plt
from model.optimizers import Adam, SGD, RMSProp
from model.metrics import calculate_recall, calculate_precision, calculate_f1, confusion_matrix

def main():
    df = preprocess_data_from_path('../data/data.csv')
    
    X = df.values[:, 1:]
    y = df['diagnosis'].values
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)


    nn = NeuralNetwork(loss_function=LossBinaryCrossEntropy())
    nn.optimizer = SGD(learning_rate=0.005, momentum=0.9, decay=1e-3)
    # nn.optimizer = SGD(learning_rate=0.0005) # vanillia SGD
    # nn.optimizer = Adam(learning_rate=0.0003, decay=1e-3)
    # nn.optimizer = RMSProp(learning_rate=0.001, rho=0.9, epsilon=1e-7, decay=0.01)
    nn.add(Dense(n_inputs=X_train.shape[1], n_neurons=16, activation=ReLU(), initializer='He'))
    nn.add(Dense(n_inputs=16, n_neurons=16, activation=Sigmoid(), initializer='Xavier'))
    nn.add(Dense(n_inputs=16, n_neurons=2, activation=Softmax(), initializer='Xavier'))
    
    print("Training...") 
    history = nn.train(X_train, y_train, n_epochs=50, batch_size=16, validation_data=(X_test, y_test), patience=3)
    
    print("Testing...") 
    
    y_pred = nn.predict(X_test)
    print(f"Predicted labels: {y_pred}")
    
    test_accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
    y_true = np.argmax(y_test, axis=1)
    recall = calculate_recall(y_pred, y_true)
    precision = calculate_precision(y_pred, y_true)
    f1 = calculate_f1(y_pred, y_true)
    print("*"*21)
    print(f"Log Loss: {history['train_loss'][-1]:.5f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("*"*21)
    
    
    plot_metrics(history)
    
    confusion_matrix(y_pred, y_true)

if __name__ == '__main__':
    main()