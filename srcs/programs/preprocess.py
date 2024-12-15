import argparse
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.preprocess import preprocess_data_from_df, train_test_split

def save_to_csv(X, y, file_path):
    """Utility function to save a dataset to a CSV file."""
    data = np.hstack((X, y.reshape(-1, 1))) 
    pd.DataFrame(data).to_csv(file_path, index=False)
    print(f"\033[92m> saved\033[0m dataset to {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing Program")
    parser.add_argument("--input", type=str, required=True, help="Path to the input dataset (CSV format).")
    parser.add_argument("--output_train", type=str, default="data/train.csv", help="Path to save the training dataset. (default: data/train.csv)")
    parser.add_argument("--output_valid", type=str, default="data/validation.csv", help="Path to save the validation dataset. (default: data/validation.csv)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting (default: 42).")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to use for validation (default: 0.2).")
    parser.add_argument("--split", type=int, default=1, help="Split the dataset into training and validation sets (default: 1)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = preprocess_data_from_df(df)

    X = df.iloc[:, 1:].values
    y = df['diagnosis'].values
    
    if args.split:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
        save_to_csv(X_train, y_train, args.output_train)
        save_to_csv(X_valid, y_valid, args.output_valid)
    else:
        save_to_csv(X, y, args.output_train)
        

if __name__ == "__main__":
    main()
