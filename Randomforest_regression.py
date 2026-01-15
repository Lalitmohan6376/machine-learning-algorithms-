from sklearn.ensemble import RandomForestRegressor

X = [[1],[2],[3],[4],[5]]
y = [2,4,6,8,10]
model = RandomForestRegressor(n_estimators=50,random_state=0)

model.fit(X,y)
pre = model.predict([[6]])
print(pre)