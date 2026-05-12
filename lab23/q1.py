import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1.
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
features = ["sepal length (cm)", "sepal width (cm)"]

# 2. Add Noise
noise = 0.2
np.random.seed(42)
df[features] = df[features] + np.random.normal(0, noise, df[features].shape)

# 3. Discretize
df['L_bin'] = pd.cut(df['sepal length (cm)'], bins=5, labels=False)
df['W_bin'] = pd.cut(df['sepal width (cm)'], bins=5, labels=False)
"""Here if the no. of bins is increased the accuracy of 
Joint prob distribution approach will increase"""

# 4. Joint Probability "Model" (Lookup Table)
# We use a group-by then find the mode for each combination
jpd_lookup = df.groupby(['L_bin', 'W_bin'])['target'].agg(lambda x: x.mode()[0] if not x.empty else np.nan)

# 5. Prediction Function
def predict_simple(row):
    try:
        return jpd_lookup.loc[row['L_bin'], row['W_bin']]
    except KeyError:
        return df['target'].mode()[0] # Fallback to global most frequent class

df['jpd_prediction'] = df.apply(predict_simple, axis=1)
jpd_acc = accuracy_score(df['target'], df['jpd_prediction'])

# 6. Decision Tree
X = df[features]
y = df['target']
dt_model = DecisionTreeClassifier(max_depth=2, random_state=42)
dt_model.fit(X, y)
y_pred_dt = dt_model.predict(X)
dt_acc = accuracy_score(y, y_pred_dt)


print("--- Results ---")
print(f"JPD Method Accuracy:      {jpd_acc:.2%}")
print(f"Decision Tree Accuracy:   {dt_acc:.2%}")