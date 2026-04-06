from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


def main():
    [X,y]=load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(random_state=9),
    n_estimators=10, # number of weak learners
    learning_rate=1.0,
    random_state=9)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("classification report:\n", class_report)

if __name__ == '__main__':
    main()