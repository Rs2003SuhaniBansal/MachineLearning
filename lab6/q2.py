import numpy as np
from sklearn.preprocessing import MinMaxScaler

"""in the array below the values are not in a range of 0 to 1
 and hence the data is normalized to fit in a common scale"""
# val = np.array([10, 12, 15, 8])
"""the values below are already in a scale of 0 to 1
 and hence the normalization will return the same array back"""
val = np.linspace(0,1)

val = val.reshape(-1,1)
"""MinMaxScaler is used to normalize the data in scikit learn"""
scaler = MinMaxScaler()
scaled_val = scaler.fit_transform(val)
# print(scaled_val)

"""MinMaxScaler from scratch - the formula behind normalization of data"""
for i in val:
    x_old = i
    x_min = min(val)
    x_max = max(val)
    x_new = (x_old - x_min) / (x_max -x_min)
    print(x_new)