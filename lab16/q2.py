import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE

def main():
    def xgBoostClassifier():
        df = pd.read_csv("Weekly.csv")
        X = df.drop(["Direction", "Today", "Year"], axis=1)
        y = df["Direction"]

        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBClassifier(n_estimators=80, learning_rate=0.02, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:",accuracy)

        f1 = f1_score(y_test, y_pred)
        print("F1 score for xgBoostClassifier",f1)

    xgBoostClassifier()

    def xgBoostRegressor():
        df = pd.read_csv("Boston.csv")
        X = df.drop(["medv"], axis=1)
        y = df["medv"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.07, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print("R2 score:",r2)
    xgBoostRegressor()

if __name__ == "__main__":
    main()




