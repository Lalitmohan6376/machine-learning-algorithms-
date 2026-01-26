from sklearn.mixture import GaussianMixture
import numpy as np

X = np.array([
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2],
    [10, 4],
    [10, 0]
])

gmm = GaussianMixture(n_components=2, random_state=42)
labels = gmm.fit_predict(X)

print(labels)
