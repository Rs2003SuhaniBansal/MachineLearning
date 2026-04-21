import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset #dataset form R package
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset (NO trailing space)
USArrests = get_rdataset('USArrests').data

print(USArrests.head())
print(USArrests.columns)
print(USArrests.mean())

"""to get the variance of the columns"""
print(USArrests.var())

# Standardize data
scaler = StandardScaler(with_std=True, with_mean=True)
USArrests_scaled = scaler.fit_transform(USArrests)

# PCA
pcaUS = PCA()
pcaUS.fit(USArrests_scaled)

scores = pcaUS.transform(USArrests_scaled)

# ---- BIPLOT ----
i, j = 0, 1
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(scores[:, i], scores[:, j])
ax.set_xlabel(f'PC{i+1}')
ax.set_ylabel(f'PC{j+1}')

for k in range(pcaUS.components_.shape[1]):
    ax.arrow(0, 0,
             pcaUS.components_[i, k],
             pcaUS.components_[j, k])
    ax.text(pcaUS.components_[i, k],
            pcaUS.components_[j, k],
            USArrests.columns[k])

plt.show()

# Flip axis
scores[:, 1] *= -1
pcaUS.components_[1] *= -1

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(scores[:, i], scores[:, j])

ax.set_xlabel(f'PC{i+1}')
ax.set_ylabel(f'PC{j+1}')

for k in range(pcaUS.components_.shape[1]):
    ax.arrow(0, 0,
             2 * pcaUS.components_[i, k],
             2 * pcaUS.components_[j, k])
    ax.text(2 * pcaUS.components_[i, k],
            2 * pcaUS.components_[j, k],
            USArrests.columns[k])

plt.show()

# Variance info
print("Std:", scores.std(axis=0, ddof=1))
print("Explained variance:", pcaUS.explained_variance_)
print("Explained ratio:", pcaUS.explained_variance_ratio_)

# ---- SCREE PLOT ----
ticks = np.arange(pcaUS.n_components_) + 1

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(ticks, pcaUS.explained_variance_ratio_, marker='o')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Proportion of Variance Explained')
axes[0].set_ylim([0, 1])
axes[0].set_xticks(ticks)

axes[1].plot(ticks, pcaUS.explained_variance_ratio_.cumsum(), marker='o')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Cumulative Proportion of Variance Explained')
axes[1].set_ylim([0, 1])
axes[1].set_xticks(ticks)

plt.show()

# Example
a = np.array([1, 2, 8, -3])
print(np.cumsum(a))


