import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X =  [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y = 0,0,0,0,0,1,1,1,1,1

model = LogisticRegression()
model.fit(X,y)
y_pred = model.predict(X)
cm = confusion_matrix(y,y_pred)
print("Actual label:", y)
print("predicated labels:", y_pred)
print("\n confusion matrix:")
print(cm)
