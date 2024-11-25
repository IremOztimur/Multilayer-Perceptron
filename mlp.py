import pandas as pd
from srcs.preprocess import get_preprocessed_data, train_test_split, to_categorical
import srcs.model.layer as layer
from srcs.model.activation import ReLU, Softmax

if __name__ == '__main__':
    df = get_preprocessed_data()
    
    X = df.values[:, 1:]
    y = df['diagnosis'].values
    y = to_categorical(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Print the shapes to verify
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    dense1 = layer.Dense(n_inputs=X_train.shape[1], n_neurons=16)
    dense2 = layer.Dense(n_inputs=16, n_neurons=2)

    dense1.forward(X_train)
        
    print(dense1.output.shape)
    
    activation1 = ReLU()
    activation2 = Softmax()
    
    activation1.forward(dense1.output)
    
    dense2.forward(activation1.output)
    
    activation2.forward(dense2.output)
    print(activation2.output[:5])
    print(activation2.output.shape)
    


