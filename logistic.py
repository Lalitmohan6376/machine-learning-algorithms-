from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Data (0=fail,1=pass)
X = [[1],[2],[3],[4],[5]]
y = [0,0,0,1,1]

#split the data into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

pre = model.predict([[2.22]])

print(pre)
