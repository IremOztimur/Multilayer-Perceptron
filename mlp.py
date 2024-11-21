import pandas as pd
from srcs.preprocess.data import get_preprocessed_data, train_test_split
from srcs.preprocess.one_hot_encoder import to_categorical


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


