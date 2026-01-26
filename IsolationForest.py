from sklearn.ensemble import IsolationForest
import numpy as np

X = np.array([
    [1],
    [2],
    [2],
    [3],
    [100]   # outlier
])

model = IsolationForest(contamination=0.2, random_state=42)
labels = model.fit_predict(X)

print(labels)  
# -1 = anomaly, 1 = normal
