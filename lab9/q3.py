from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    sonar = fetch_openml(name='sonar', version=1, as_frame=False)
    X = sonar.data
    y = sonar.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

    model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=999)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report: \n {classification_report(y_test, y_pred)}")

    plt.figure(figsize=(16, 8))
    plot_tree(model, filled=True)
    plt.show()
    """plot tree is used to visualize how the decision tree is making decisions.
    figsize controls the size of the figure"""

if __name__ == '__main__':
    main()
