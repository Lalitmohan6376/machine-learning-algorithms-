from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

X = [
    [2,5],
    [3,6],
    [1,4],
    [8,7],
    [7,6],
    [9,8], 
    [4,5],
    [6,6],
    [5,7],
    [10,8]
]

y = [0,0,0,1,1,1,0,1,1,1]

model = GradientBoostingClassifier(n_estimators=50,learning_rate=0.1,max_depth=3,random_state=42)

model.fit(X,y)
pre = model.predict([[11,5]])
print(pre)