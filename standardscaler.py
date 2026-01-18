import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
X = np.array([[1],
                 [2],
                 [3],
                 [4],
                 [5],
                 [6]])

y = np.array([30000,35000,40000,45000,50000,55000])

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(x_scaled,y)

new_data = np.array([[7]])
new_data_scaled = scaler.transform(new_data)
pre = model.predict(new_data_scaled)

print("scaled training data: \n",x_scaled)
print("predicated value:\n",pre[0])


plt.plot(X,y)
plt.show()
