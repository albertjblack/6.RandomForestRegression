import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("data.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(x,y)

x_opt = np.arange(min(x), max(x), 0.01)
x_opt = x_opt.reshape(len(x_opt), 1)

plt.scatter(x, y, color = "red")
plt.plot(x_opt, regressor.predict(x_opt), color = "blue")
plt.show()

prediction = regressor.predict([[6.5]])
print(prediction)