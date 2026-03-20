import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

data = load_diabetes()
X, y = data.data, data.target

"""Decision Tree Regressors are high variance models, meaning small changes
in training data (like different random_state values) can lead to different trees."""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=999
)

"""This function calculates the Mean Squared Error (MSE) of a set of target values.
In this context, it represents the variance of the node.
Lower MSE means the values are more similar (better split)."""
def mse(y):
    return np.mean((y - np.mean(y, dtype=np.float64))**2)

"""This function finds the best feature and threshold to split the data.
It tries every feature and every unique value as a threshold,
then calculates the weighted MSE of the split.
The goal is to minimize the total error after splitting."""
def best_split(X, y):
    best_feature, best_threshold = None, None
    best_error = float("inf")

    n_samples, n_features = X.shape

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])

        for threshold in thresholds:
            left = y[X[:, feature] <= threshold]
            right = y[X[:, feature] > threshold]

            """If either side of the split has no samples,
            it is not a valid split, so it's skipped."""
            if len(left) == 0 or len(right) == 0:
                continue

            """Compute weighted MSE:
            Each side contributes to total error proportional to its size."""
            error = (len(left)*mse(left) + len(right)*mse(right)) / len(y)

            """Update the best split if current split has lower error."""
            if error < best_error:
                best_error = error
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

"""This function builds the decision tree recursively.
At each step, it finds the best split and divides the data into left and right.
The recursion continues until stopping conditions are met."""
def build_tree(X, y, depth=0, max_depth=4):

    """Stopping conditions:
    1. Maximum depth reached (to prevent overfitting)
    2. Too few samples left to split meaningfully
    In such cases, return the mean value (leaf node prediction)."""
    if depth == max_depth or len(y) < 5:
        return np.mean(y)

    feature, threshold = best_split(X, y)

    """If no valid split is found, it returns a leaf node."""
    if feature is None:
        return np.mean(y)

    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    """Recursively build left and right subtrees using split data."""
    left_tree = build_tree(X[left_mask], y[left_mask], depth+1, max_depth)
    right_tree = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)

    """Each node is represented as a tuple:
    (feature index, threshold, left subtree, right subtree)"""
    return (feature, threshold, left_tree, right_tree)

"""This function predicts the output for a single input sample.
It traverses the tree from root to leaf based on feature thresholds."""
def predict_sample(x, tree):

    """If the current node is not a tuple, it is a leaf node,
    so we directly return the prediction value."""
    if not isinstance(tree, tuple):
        return tree

    feature, threshold, left, right = tree

    """Depending on the feature value, move left or right in the tree."""
    if x[feature] <= threshold:
        return predict_sample(x, left)
    else:
        return predict_sample(x, right)

"""This function applies prediction to all samples in the dataset
by calling predict_sample for each row."""
def predict(X, tree):
    return np.array([predict_sample(x, tree) for x in X])

tree = build_tree(X_train, y_train)

y_pred = predict(X_test, tree)

mse_value = np.mean((y_test - y_pred)**2)
print("MSE:", mse_value)