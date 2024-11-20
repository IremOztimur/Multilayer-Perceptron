import pandas as pd
from srcs.preprocess.eda import get_data
from srcs.preprocess.one_hot_encoder import to_categorical

if __name__ == '__main__':
    df = get_data()
    
    x = df.values[:, 1:]
    y = df['diagnosis'].values
    y = to_categorical(y)
    