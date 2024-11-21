import pandas as pd
from srcs.preprocess.eda import get_preprocessed_data
from srcs.preprocess.one_hot_encoder import to_categorical
from sklearn.preprocessing import RobustScaler


if __name__ == '__main__':
    df = get_preprocessed_data()
    
    x = df.values[:, 1:]
    y = df['diagnosis'].values
    y = to_categorical(y)
    
    

