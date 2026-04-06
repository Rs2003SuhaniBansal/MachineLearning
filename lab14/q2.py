from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def main():

    X, y = load_iris(return_X_y=True)

    """Convert to binary classification (class 0 vs others)
    here  one vs rest (OVR) multiclass classification is being used where 0 is -1 and rest (1&2) is +1
    therefore the multiclass dataset is converted to binary classification."""
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    n_samples, n_features = X_train.shape

    """Initialize weights, same for all samples initially.
    which means all samples are equally important."""
    weights = np.ones(n_samples) / n_samples

    n_estimators = 10
    alphas = []
    stumps = []

    """AdaBoost loop"""
    for _ in range(n_estimators):

        best_feature = None
        best_split = None
        best_polarity = 1
        min_error = float("inf")

        """Try all features to find the best decision stump."""
        for feature in range(n_features):

            values = np.sort(np.unique(X_train[:, feature]))
            splits = (values[:-1] + values[1:]) / 2

            for split in splits:

                predictions = np.ones(n_samples)
                predictions[X_train[:, feature] < split] = -1

                error = np.sum(weights[y_train != predictions])

                """Here polarity is flipped if error > 0.5, 
                that is, if classifier is worse than random prediction is flipped"""
                polarity = 1
                if error > 0.5:
                    error = 1 - error
                    polarity = -1


                """This gives the best weak learner/stump"""
                if error < min_error:
                    min_error = error
                    best_feature = feature
                    best_split = split
                    best_polarity = polarity

        """Compute alpha,
        epsilon is added in the denominator to avoid division by zero and logarithmic overflow"""
        epsilon = 1e-10
        alpha = 0.5 * np.log((1 - min_error) / (min_error + epsilon))

        """Store stump"""
        stumps.append((best_feature, best_split, best_polarity))
        alphas.append(alpha)

        """Update weights,
        Here weight for misclassified samples is high and for correctly classified samples it's low."""
        predictions = np.ones(n_samples)
        if best_polarity == 1:
            predictions[X_train[:, best_feature] < best_split] = -1
        else:
            predictions[X_train[:, best_feature] >= best_split] = -1

        weights *= np.exp(-alpha * y_train * predictions)
        weights /= np.sum(weights)

    """Final prediction"""
    final_pred = np.zeros(len(X_test))

    for alpha, stump in zip(alphas, stumps):
        feature, split, polarity = stump

        pred = np.ones(len(X_test))
        if polarity == 1:
            pred[X_test[:, feature] < split] = -1
        else:
            pred[X_test[:, feature] >= split] = -1

        final_pred += alpha * pred

    y_pred = np.sign(final_pred)

    acc = accuracy_score(y_test, y_pred)
    print("AdaBoost Accuracy:", acc)

if __name__ == "__main__":
    main()