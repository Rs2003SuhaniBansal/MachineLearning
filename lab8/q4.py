from numpy import mean
from numpy import std
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from pandas import read_csv
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
# load the dataset
dataset = read_csv(url, header=None)
# retrieve the array of data

data = dataset.values
# separate into input and output columns

X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)

# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

def ordinal_encoder(X_train, y_train, X_test, y_test):
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    """here, in breast_cancer dataset there is a value '24-26' in column 3 of X_test
    which is not present in the train set. Therefore it raises an error of unknown value.
    This error can be handled by using handle_unknown and unknown_value in the OrdinalEncoder function."""
    ordinal_encoder.fit(X_train)
    X_train = ordinal_encoder.fit_transform(X_train)
    X_test = ordinal_encoder.transform(X_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)

    accuracy = accuracy_score(y_test, yhat)
    print("Accuracy:%.2f" % (accuracy * 100))

def one_hot_encoder(X_train, y_train, X_test, y_test):
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    onehot_encoder.fit(X_train)
    X_train = onehot_encoder.transform(X_train)
    X_test = onehot_encoder.transform(X_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    yhat = model.predict(X_test)

    accuracy = accuracy_score(y_test, yhat)
    print("Accuracy:%.2f" % (accuracy * 100))

print("Ordinal Encoding:")
ordinal_encoder(X_train, y_train, X_test, y_test)
print()
print("One-Hot Encoding:")
one_hot_encoder(X_train, y_train, X_test, y_test)