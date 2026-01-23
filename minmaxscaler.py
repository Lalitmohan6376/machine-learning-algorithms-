from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Dataset load
data = load_iris()
X = data.data
y = data.target

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. MinMaxScaler apply
scaler = MinMaxScaler(feature_range=(0, 1))

# 4. Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# 5. Transform test data
X_test_scaled = scaler.transform(X_test)

# 6. Output check
print("Before Scaling (first row):")
print(X_train[0])

print("\nAfter Scaling (first row):")
print(X_train_scaled[0])
