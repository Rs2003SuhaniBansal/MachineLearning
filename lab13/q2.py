import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = load_diabetes(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9
)


"""number of decision tree models to train.
this simulates an ensemble (bagging-like approach)."""
n_estimators = 50

"""this list will store trained models."""
models = []

"""train multiple models using bootstrap sampling.
each model is trained on a random sample (with replacement)
from the training dataset."""
for i in range(n_estimators):

    """generate bootstrap sample indices.
    replace=True allows repeated sampling."""
    indices = np.random.choice(len(X_train), len(X_train), replace=True)

    """create sampled dataset using generated indices."""
    X_sample = X_train[indices]
    y_sample = y_train[indices]

    """train Decision Tree Regressor on sampled data."""
    model = DecisionTreeRegressor()
    model.fit(X_sample, y_sample)

    """store trained model in list."""
    models.append(model)

"""store predictions from each model."""
predictions = []

"""generate predictions using all trained models."""
for model in models:
    pred = model.predict(X_test)
    predictions.append(pred)

"""convert list of predictions to numpy array
and take average across all models (ensemble prediction)."""
predictions = np.array(predictions)
y_pred = np.mean(predictions, axis=0)


mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)