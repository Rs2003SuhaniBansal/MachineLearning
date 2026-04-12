from sklearn.tree import DecisionTreeRegressor
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.2, 1.8, 3.0, 3.6, 5.1])

""" This list is to store trained decision trees"""
trees = []

"""Train multiple trees (ensemble)"""
for i in range(5):
    """Bootstrap sampling: randomly select samples with replacement
    This ensures each tree is trained on slightly different data"""
    indices = np.random.choice(len(X), len(X), replace=True)

    """Create bootstrap dataset"""
    X_sample = X[indices]
    y_sample = y[indices]

    """Create decision tree regressor with depth limit"""
    tree = DecisionTreeRegressor(max_depth=2)

    """Train tree on bootstrap sample"""
    tree.fit(X_sample, y_sample)

    """Store trained tree"""
    trees.append(tree)


"""Function to aggregate predictions from all trees"""
def aggregate_predictions(trees, X_test):
    """Get predictions from each tree
    Shape will be: (number of trees, number of test samples)"""
    predictions = np.array([tree.predict(X_test) for tree in trees])

    """Average predictions across trees (column-wise)
    axis=0 means averaging predictions for each test sample"""
    return np.mean(predictions, axis=0)


X_test = np.array([[2.5], [3.5]])

final_prediction = aggregate_predictions(trees, X_test)

print("Final Prediction:", final_prediction)