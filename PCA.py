from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 1. Dataset load
data = load_iris()
X = data.data

# 2. PCA apply (2 dimensions me reduce)
pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X)

# 3. Output
print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)

print("\nFirst 5 reduced values:")
print(X_reduced[:5])

# 4. Explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)
