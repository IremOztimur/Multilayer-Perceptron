import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler


columns = [
    'id', 'diagnosis', 
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 
    'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
    'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 
    'fractal_dimension_worst'
]


def get_preprocessed_data():
    file_path = os.path.join(os.path.dirname(__file__), '../../data/data.csv')
    df = pd.read_csv(file_path, header=None, names=columns)
    scaler = RobustScaler()
    
    df = df.drop('id', axis=1)
    df["diagnosis"] = df["diagnosis"].map({'M': 1, 'B': 0})
    
    x = df.values[:, 1:]

    scaled_data = scaler.fit_transform(x)
    df.iloc[:, 1:] = scaled_data
    
    df = drop_weak_correlation(df, 'diagnosis', threshold=0.1)

    return df

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    split_index = int((1 - test_size) * X.shape[0])
    X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:]
    y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:]

    return X_train, X_test, y_train, y_test

def to_categorical(x, num_classes=None):
    """ Converts a class vector (integers) to binary class matrix.
    
        Args:
            x (numpy array): class vector to be converted.
            num_classes (int): total number of classes.
        
        Returns:
            A binary matrix representation of the input as a NumPy array. The class axis is placed last.
    
    """
    if not num_classes:
        num_classes = np.max(x) + 1
    one_hot_labels = np.zeros((x.size, num_classes))
    one_hot_labels[np.arange(x.size), x] = 1
    
    return one_hot_labels

def drop_weak_correlation(df, target_column, threshold=0.1):
    """
    Drops columns from df that have weak correlation with the target column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The column name of the target variable.
    threshold (float): The minimum absolute correlation required to keep a column.

    Returns:
    pd.DataFrame: A new DataFrame with only columns that have strong correlation with the target.
    """
    correlation_matrix = df.corr()

    target_correlation = correlation_matrix[target_column]
    
    weakly_correlated_features = target_correlation[abs(target_correlation) < threshold].index

    df_reduced = df.drop(columns=weakly_correlated_features)
    
    print(f"Dropped columns with weak correlation: {list(weakly_correlated_features)}")
    
    return df_reduced

if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), '../../data/data.csv')
    df = pd.read_csv(file_path, header=None, names=columns)

    df = df.drop('id', axis=1)
    df["diagnosis"] = df["diagnosis"].map({'M': 1, 'B': 0})
    
    x = df.values[:, 1:]
    y = df['diagnosis'].values
    y = to_categorical(y)
    
    for i in range(x.shape[1]):
        plt.figure()
        plt.hist(x[:, i], bins=30)
        plt.title(f'Feature {i}')
        plt.show()
    
    print("Shape: ", df.shape)
    print("*"*100)
    print(df.head())
    print("*"*100)
    print(df.describe())
    print("*"*100)
    print(df.info())
    print("*"*100)
    print(df.isna().any().any())