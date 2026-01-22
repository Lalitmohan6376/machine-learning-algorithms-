from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([[2],[4],[9],[1],[0]])

model = PolynomialFeatures()
x = model.fit_transform(X)

print(x)