import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

def main():
    def Gradient_Boosting_Regressor():
        df = pd.read_csv("Boston.csv", index_col=0)
        # print(df.head())
        X = df.drop(["medv"], axis=1)
        y = df["medv"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        gradient_boost_reg = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
        gradient_boost_reg.fit(X_train, y_train)
        pred_y = gradient_boost_reg.predict(X_test)

        train_pred = gradient_boost_reg.predict(X_train)
        r2 = r2_score(y_train, train_pred)
        print("train r2:", r2)
        # train_mse = mean_squared_error(y_train, train_pred)

        r2 = r2_score(y_test, pred_y)
        mse = mean_squared_error(y_test, pred_y)
        print("MSE of Gradient Boosting Regressor:",mse)
        print("Accuracy:", r2)
        print()

    def Gradient_Boosting_Classifier():
        df = pd.read_csv("Weekly.csv")

        """The 'Today' column may directly encode the answer leading to overfitting of the model. 
        Therefore that column is dropped"""
        X = df.drop(["Direction", "Today"], axis=1)
        y = df["Direction"]

        le = LabelEncoder()
        y = le.fit_transform(df["Direction"])

        # print(df.isnull().sum())
        # print(df.describe())
        # print(df.info())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # print(df.head())

        gradient_boost_clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.01,
            min_samples_split=5,
            random_state=42)
        gradient_boost_clf.fit(X_train, y_train)
        pred_y = gradient_boost_clf.predict(X_test)

        train_pred = gradient_boost_clf.predict(X_train)
        print("Train Accuracy:", accuracy_score(y_train, train_pred))

        accuracy = accuracy_score(y_test, pred_y)
        print("Accuracy of Gradient Boosting Classifier:",accuracy)

        cross_val = cross_val_score(gradient_boost_clf, X, y, cv=10, n_jobs=-1)
        print("CV Accuracy:", cross_val.mean())

        f1 = f1_score(y_test, pred_y)
        print("F1 Score of Gradient Boosting Classifier:",f1)

    Gradient_Boosting_Regressor()
    Gradient_Boosting_Classifier()

if __name__ == "__main__":
    main()