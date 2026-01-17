from sklearn.preprocessing import LabelEncoder,MinMaxScaler
X = ["female","male"]
le = LabelEncoder()
v = le.fit_transform(X)
print(v)

scale = MinMaxScaler()
X = [[1000],[2000],[3000],[4000],[5000],[6000]]
y = [2,3,4,5,6]
x_scaled = scale.fit_transform(X,y)
print(x_scaled)