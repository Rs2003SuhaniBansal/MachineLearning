import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score

# Wisconsin Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

""" Standardization is important for regularization
because regularization penalizes the size of coefficients,
features must be on the same scale (mean=0, variance=1)."""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=999)

"""Build Ridge Classifier (L2 Regularization)
alpha: equivalent of lambda (not used because it is a reserved keyword in python) 
Regularization strength (higher = more penalty)"""
ridge_model = RidgeClassifier(alpha=1.0)
ridge_model.fit(X_train, y_train)

"""Build Lasso Classifier (L1 Regularization)
# LogisticRegression with the 'liblinear' solver is used to support L1"""
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
lasso_model.fit(X_train, y_train)

# Evaluation
ridge_pred = ridge_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)

print("Ridge Classifier (L2):")
print(f"Accuracy: {accuracy_score(y_test, ridge_pred):.4f}")
print()
print("Lasso Classifier (L1):")
print(f"Accuracy: {accuracy_score(y_test, lasso_pred):.4f}")