from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2],
    [10, 4],
    [10, 0]
])

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
labels = gmm.fit_predict(X)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Gaussian Mixture Model Clustering")
plt.show()

print("Cluster Labels:", labels)
