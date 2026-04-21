import numpy as np

np.random.seed(10)
n = 100

# Simulated features
BP = np.random.normal(loc=80, scale=5, size=n)
Age = np.random.randint(20, 60, size=n)
Cholesterol = np.random.normal(200, 20, size=n)

# Stack features into matrix (n × 3)
X = np.column_stack((BP, Age, Cholesterol))

# Simulated regression target
y = 0.4*BP + 0.2*Age + 0.1*Cholesterol + np.random.normal(0, 5, n)


# Recursive Tree Function
def build_tree(X, y, min_samples=10, depth=0, max_depth=3):

    n_samples, n_features = X.shape

    # Stopping condition
    if n_samples <= min_samples or depth >= max_depth:
        return np.mean(y)

    best_error = float("inf")
    best_feature = None
    best_split = None

    # Try all features
    for feature_index in range(n_features):

        values = X[:, feature_index]
        possible_splits = np.unique(values)

        # Try all possible split points
        for split in possible_splits:

            left_mask = values <= split
            right_mask = values > split

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            # Compute squared error
            error = np.sum((y_left - np.mean(y_left))**2) + \
                    np.sum((y_right - np.mean(y_right))**2)

            if error < best_error:
                best_error = error
                best_feature = feature_index
                best_split = split

    if best_feature is None:
        return np.mean(y)

    # Partition data
    left_mask = X[:, best_feature] <= best_split
    right_mask = X[:, best_feature] > best_split

    # Recursive calls
    left_tree = build_tree(X[left_mask], y[left_mask],
                           min_samples, depth+1, max_depth)

    right_tree = build_tree(X[right_mask], y[right_mask],
                            min_samples, depth+1, max_depth)

    return {
        "feature": best_feature,
        "split": best_split,
        "left": left_tree,
        "right": right_tree
    }


tree = build_tree(X, y)
print(tree)