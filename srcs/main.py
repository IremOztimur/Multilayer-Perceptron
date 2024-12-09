from model.neural_network import NeuralNetwork
from model.layer import Dense
from model.activation import ReLU, Softmax, Sigmoid
from model.loss import LossCategoricalCrossEntropy, LossBinaryCrossEntropy
import numpy as np
from model.preprocess import get_preprocessed_data, train_test_split, to_categorical
from visualizer import plot_metrics, plot_step_loss
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = get_preprocessed_data()
    
    X = df.values[:, 1:]
    y = df['diagnosis'].values
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)


    nn = NeuralNetwork(learning_rate=0.0005, loss_function=LossCategoricalCrossEntropy())
    nn.add(Dense(n_inputs=X_train.shape[1], n_neurons=16, activation=ReLU(), initializer='He'))
    nn.add(Dense(n_inputs=16, n_neurons=8, activation=Sigmoid(), initializer='Xavier'))
    nn.add(Dense(n_inputs=8, n_neurons=2, activation=Softmax(), initializer='Xavier'))
    
    print("Training...") 
    history = nn.train(X_train, y_train, n_epochs=20, batch_size=8, validation_data=(X_test, y_test), patience=3)
    
    print("Testing...") 
    
    y_pred = nn.predict(X_test)
    print(f"Predicted labels: {y_pred}")
    
    test_accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
    print("*"*21)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print("*"*21)
    
    plot_metrics(history)
    # plot_step_loss(history['step_loss'])
    
    nn.save_model('depo/saved_model.npy')


