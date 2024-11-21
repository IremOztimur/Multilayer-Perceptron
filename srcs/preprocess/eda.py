import pandas as pd
import os
import matplotlib.pyplot as plt
from one_hot_encoder import to_categorical
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
    
    return df


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