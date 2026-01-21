from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

X = [[1,8],[2,7],[3,6],[4,6],[5,5],[6,5],[7,4],[8,3]]

X_train,X_test = train_test_split(X,test_size=0.25,random_state=42)
model = KMeans(n_clusters=2,random_state=42)
model.fit(X_train)
pre = model.predict([[3,9]])
print(pre)
