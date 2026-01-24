from sklearn.cluster import MeanShift

X = [
    [1, 2], [2, 3], [3, 4],
    [8, 7], [8, 8], [25, 80]
]

model = MeanShift()
labels = model.fit_predict(X)

print(labels)
