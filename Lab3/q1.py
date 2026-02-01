import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def main():
    #Loading the data
    data = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
    # print(data.head())

    #Dividing it into training set and test set
    X = data.drop(columns=["disease_score"])
    y = data["disease_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)

    model = LinearRegression()

    model.fit(X_train_scale, y_train)

    pred_y = model.predict(X_test_scale)

    # print r2 score
    r2 = r2_score(y_test, pred_y)
    print("R² Score:", r2)


if __name__ == '__main__':
    main()
