from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from ISLP import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


data = load_data("NCI60")
X = data['data']
y = np.ravel(data['labels'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- PCA Approach ----
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

clf_pca = LogisticRegression(C=5, penalty='l2', max_iter=10000)
clf_pca.fit(X_train_pca, y_train)
pca_acc = clf_pca.score(X_test_pca, y_test)

# ---- Hierarchical Clustering Approach ----
clustering = AgglomerativeClustering(n_clusters=50)
gene_clusters = clustering.fit_predict(X_train.T)

# Creates cluster features
def reduce_features(X, clusters):
    return np.array([X[:, clusters == i].mean(axis=1)
                     for i in np.unique(clusters)]).T

X_train_hc = reduce_features(X_train, gene_clusters)
X_test_hc = reduce_features(X_test, gene_clusters)

clf_hc = LogisticRegression(C=5, penalty='l2', max_iter=10000)
clf_hc.fit(X_train_hc, y_train)
hc_acc = clf_hc.score(X_test_hc, y_test)

print("PCA Accuracy:", pca_acc)
print("HC Accuracy:", hc_acc)


cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

scores_pca = cross_val_score(
    clf_pca,
    X_train_pca,
    y_train,
    cv=cv
)

scores_hc = cross_val_score(
    clf_hc,
    X_train_hc,
    y_train,
    cv=cv
)

print("Cross-val score of PCA",scores_pca.mean())
print("cross-val score of Hierarchical clustering",scores_hc.mean())