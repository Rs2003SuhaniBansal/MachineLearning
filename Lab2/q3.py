from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
def main():

    #Loading the data
    [X, y] = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

    #Standardizing the data
    scaler=StandardScaler()
    scaler=scaler.fit(X_train)
    X_train_scale=scaler.transform(X_train)
    X_test_scale=scaler.transform(X_test)

    #initializing the model
    model = LinearRegression()

    # training a model
    model.fit(X_train_scale, y_train)


    pred_y=model.predict(X_test_scale)

    # print r2 score
    r2 = r2_score(y_test, pred_y)
    print("R² Score:", r2)


if __name__ == '__main__':
    main()
