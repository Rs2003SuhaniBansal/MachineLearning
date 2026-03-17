import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entropy function
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    """probs calculates probability of each class."""
    return -np.sum(probs * np.log2(probs))
"""here entropy is calculated and np.unique finds unique class labels.
return_counts=True returns how many times each class occurs."""

# Information Gain
def information_gain(parent, left, right):
    n = len(parent)
    n_left = len(left)
    n_right = len(right)

    if n_left == 0 or n_right == 0:
        return 0
    """if the left split or the right split is empty, it's useless.
     Therefore information gain is returned as 0"""

    parent_entropy = entropy(parent)
    child_entropy = (n_left/n)*entropy(left) + (n_right/n)*entropy(right)

    return parent_entropy - child_entropy


def best_split(X, y):
    """this function finds the best feature and threshold to split the data."""

    best_feature = None
    best_threshold = None
    best_gain = -1
    """here the variables are initialized to store the best split found so far."""

    n_samples, n_features = X.shape

    for feature in range(n_features):
        """here we are looping through every feature."""

        thresholds = np.unique(X[:, feature])
        """all unique values of that feature are found 
        and are tested as a possible split threshold"""

        for threshold in thresholds:

            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold
            """here every value of the sample is compared with the threshold and a boolean array is created"""

            y_left = y[left_mask]
            y_right = y[right_mask]

            gain = information_gain(y, y_left, y_right)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
            """the new best split is stored."""

    return best_feature, best_threshold


# Build tree recursively
def build_tree(X, y, depth=0, max_depth=4):

    classes, counts = np.unique(y, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    """majority_class finds the most frequent class."""

    # Stopping conditions
    if len(classes) == 1 or depth == max_depth or len(y) < 5:
        return majority_class
    """if only one class is present, the tree has reached max depth.
    it is returned as the leaf node."""
    feature, threshold = best_split(X, y)

    if feature is None:
        return majority_class
    """if no valid spit exists then the leaf is returned."""

    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    left_tree = build_tree(X[left_mask], y[left_mask], depth+1, max_depth)
    right_tree = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)
    """right subtree and left subtree are built."""

    return (feature, threshold, left_tree, right_tree)


# Predict single sample
def predict_sample(x, tree):
    """class for a single sample is predicted."""

    if not isinstance(tree, tuple):
        return tree
    """if node is not a tuple, it is a leaf node.
    the class label is returned."""

    feature, threshold, left, right = tree

    if x[feature] <= threshold:
        return predict_sample(x, left)
    else:
        return predict_sample(x, right)


# Predict dataset
def predict(X, tree):
    return np.array([predict_sample(x, tree) for x in X])
"""here, the labels for the entire dataset are predicted and returned."""


# Train model
tree = build_tree(X_train, y_train)
"""here decision tree for training data is built."""

# Predictions
y_pred = predict(X_test, tree)
"""classes for test data are predicted."""

# Accuracy
accuracy = np.mean(y_pred == y_test)

print("Accuracy:", accuracy)