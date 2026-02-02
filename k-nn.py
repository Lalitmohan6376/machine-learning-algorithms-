from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder

X = [[40,20],[50,50],[60,90],[10,25],[70,70]]
y = ['red','red','blue','blue','red']
le = LabelEncoder()
model = KNeighborsClassifier()
y = le.fit_transform(y)
print(y)
model.fit(X,y)
def fun():
    pre =  model.predict([[48,72]])
    return pre

result = fun()
if result == 0:
    print("Blue")
else:

    print("Red")
