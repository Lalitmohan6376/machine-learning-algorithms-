from sklearn.cluster import AgglomerativeClustering

X = [
    [1, 2], [2, 3], [3, 4],
    [8, 7], [8, 8], [25, 80]
]

model = AgglomerativeClustering(n_clusters=2)
labels = model.fit_predict(X)

print(labels)
