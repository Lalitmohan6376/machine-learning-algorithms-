from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Example data (hours studied vs marks)
X = [[1],[2],[3],[4],[5]]
y = [10,20,30,40,50]

#split data(80% traning, 20% testing)
X_train,X_test,y_train,y_text = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

pre = model.predict(X_test)
print(pre[0])