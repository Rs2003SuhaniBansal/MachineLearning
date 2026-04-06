from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import pandas as pd

def main():
    def decision_tree_regressor():
        data = load_diabetes()
        df = pd.concat([pd.DataFrame(data.data), pd.DataFrame(data.target)], axis=1)
        X = data.data
        y = data.target

        print(df.head())

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=9
        )

        # Bagging Regressor
        bag_reg = BaggingRegressor(
            estimator=DecisionTreeRegressor(),
            max_samples=0.5,
            n_estimators=90,
            bootstrap=True,
            random_state=999,
            n_jobs=-1
        )
        """max_samples builds trees of varying data as it takes only some part of the data 
        (here, 60% of the data is taken) this leads to every tree being different from the other.
        n_estimators is the number of trees that have to be built.
        bootstrap=True will take data with replacement and hence values of samples can repeat. 
        This gives better accuracy than doing bootstrap=False where data is taken without
        replacement"""

        # Train model
        bag_reg.fit(X_train, y_train)

        # Prediction
        y_pred = bag_reg.predict(X_test)

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Mean Squared Error of decision tree regressor:", mse)
        print("R2 Score of decision tree regressor:", r2)

    def descision_tree_classifier():
        data = load_iris()
        X = data.data
        y = data.target

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=9
        )

        # Base model
        base_model = DecisionTreeClassifier()
        bag_reg = BaggingClassifier(
            estimator=base_model,
            n_estimators=500,
            max_samples=0.6,
            random_state=9
        )
        bag_reg.fit(X_train, y_train)
        y_pred = bag_reg.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy of decision tree classifier:", accuracy)

    decision_tree_regressor()
    descision_tree_classifier()

if __name__ == "__main__":
    main()
