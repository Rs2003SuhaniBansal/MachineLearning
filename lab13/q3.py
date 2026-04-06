from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

def main():
    def random_forest_regressor():
        # Load dataset
        [X, y] = load_diabetes(return_X_y=True)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=9)

        # Create Random Forest Regressor model
        rf_reg = RandomForestRegressor(
            n_estimators=120,
            random_state=9)

        # Train the model
        rf_fit = rf_reg.fit(X_train, y_train)

        # Predict on test data
        y_pred = rf_reg.predict(X_test)

        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Mean Squared Error:", mse)
        print("R2 Score:", r2)

    def random_forest_classifier():
        [X, y] = load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
        rf_clf = RandomForestClassifier(n_estimators=5,random_state=9)

        rf_clf.fit(X_train, y_train)

        y_pred = rf_clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

    random_forest_regressor()
    random_forest_classifier()

if __name__ == "__main__":
    main()
