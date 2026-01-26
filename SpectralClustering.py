from sklearn.cluster import SpectralClustering
import numpy as np

X = np.array([
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2],
    [10, 4],
    [10, 0]
])

model = SpectralClustering(
    n_clusters=2,
    affinity='nearest_neighbors',
    random_state=42
)

labels = model.fit_predict(X)
print(labels)
