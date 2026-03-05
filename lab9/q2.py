import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(10)
n = 100 #number of samples

"""np.random.normal takes (mean, standard_deviation, size) as input
np.random.randint takes (low, high, size) as input
This overall creates a simulated dataset for regression.
here age is discrete data, cholesterol and BP and continuous data.
Loc mean location parameter which shifts the distribution left or right
scale stretches or shrinks the distribution."""
BP = np.random.normal(80, 5, n)
Age = np.random.randint(20, 60, n)
Cholesterol = np.random.normal(200, 20, n)

"""column_stack is used to change the shape of the data, from (100,) it becomes (100,3) 
where 100 is the number of samples"""
X = np.column_stack((BP, Age, Cholesterol))

y = 0.5*BP + 0.3*Age + 0.1*Cholesterol + np.random.normal(0, 5, n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

model = DecisionTreeRegressor(max_depth=3, min_samples_split=5, random_state=999)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

#To plot the tree structure
plt.figure(figsize=(12,6))
plot_tree(model, filled=True)
plt.show()