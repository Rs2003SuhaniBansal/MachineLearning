import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def question1():

    # Load dataset
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)

    # Split dataset
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

    # Compute mean and std using TRAIN data only
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)

    # Standardize training data
    X_train_scaled = (X_train - train_mean) / train_std

    # Standardize test data using SAME values
    X_test_scaled = (X_test - train_mean) / train_std

    print("Train standardized data (first 5 rows):")
    print(X_train_scaled.head())

    print("\nTest standardized data (first 5 rows):")
    print(X_test_scaled.head())


def main():
    question1()


if __name__ == "__main__":
    main()