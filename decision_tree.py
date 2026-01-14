from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Data (Stduy hours, sleep hours) -> pass(1),fail(0)
X = [[1,8],[2,7],[3,6],[4,6],[5,5],[6,5],[7,4],[8,3]]
y = [0,0,0,1,1,1,1,1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
pre = model.predict([[3,8]])
print(pre)