from sklearn.cluster import DBSCAN

# sample data
X = [
    [1, 2], [2, 3], [2, 2],
    [8, 7], [8, 8], [25, 80]
]

model = DBSCAN(eps=3, min_samples=2)
labels = model.fit_predict(X)

print(labels)
