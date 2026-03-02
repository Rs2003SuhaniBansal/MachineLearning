"""Data Standardization / Z-Score Normalization"""
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
[values, y] = fetch_california_housing(return_X_y=True)

"""mean and standard deviation of unstandardized values of the features is calculated"""
mean = np.mean(values)
std_dev = np.std(values)

values_new = [] # created to append rows
for val in values:
    val_new = [] # created to append columns
    for i in val:
        x_old = i
        x_new = (x_old - mean) / std_dev
        """float fixes the printing of val_new,
        round: rounds off the float values (here till 2 decimal places)"""
        val_new.append(float(round(x_new, 2)))
    values_new.append(val_new)

"""To print the first few standardized values"""
# vals = pd.DataFrame(values_new)
# print(f'Standardized values are: \n {vals.head()}')

"""Gives the new mean and new standard deviation of the features after standardization"""
new_mean = np.mean(values_new) #output = 0
new_std = np.std(values_new) #output = 1

"""the mean and standard deviation are rounded off and printed"""
print(f"mean: {round(new_mean)}, std dev: {round(new_std)}")