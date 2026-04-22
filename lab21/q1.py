import numpy as np

"""Step 1: Random assignment"""
def random_cluster_assignment(X, k):
    return np.random.randint(0, k, size=X.shape[0])

"""Step 2: Compute centroids"""
def compute_centroids(X, labels, k):
    centroids = []

    for i in range(k):
        cluster_points = X[labels == i]

        if len(cluster_points) == 0:
            # reinitialize randomly
            centroids.append(X[np.random.randint(0, len(X))])
        else:
            centroids.append(cluster_points.mean(axis=0))

    return np.array(centroids)

"""Step 3: Assign to nearest centroid"""
def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

"""Full algorithm"""
def kmeans(X, k, max_iters=100):
    labels = random_cluster_assignment(X, k)

    for _ in range(max_iters):
        centroids = compute_centroids(X, labels, k)
        new_labels = assign_clusters(X, centroids)

        """stop if clusters don't change"""
        if np.array_equal(labels, new_labels):
            print("Converged")
            break

        labels = new_labels

    return labels, centroids

def main():
    X = np.array([[1, 1], [1, 2], [2, 2], [8, 8], [8, 9],[9,8]])

    labels, centroids = kmeans(X, k=2)

    print("Final labels:", labels)
    print("Final centroids:", centroids)

if __name__ == "__main__":
    main()