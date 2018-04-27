# Decision Tree Regression
""" We are going to solve the same problem of HR predicting if the employee is 
bluffing or telling the truth about his previous salary using DTR """

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
Data = pd.read_csv('Position_Salaries.csv')
X = Data.iloc[:, 1:2].values
y = Data.iloc[:, 2].values

""" Like before, no need of splitting the data into test and train sets as the data available is less"""
# Feature Scaling is not needed here
# Fitting the Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
Regressor = DecisionTreeRegressor()
Regressor.fit(X, y)

# Predicting the new result
y_pred = Regressor.predict(6.9)

# Visualizing the DTR results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

""" Without the above X_grid, the graph we see first hand is wrong, cuz between two succesive
points is an interval(a split(leaf)), and the DTR calculates the average of each leaf and 
therefore the graph should look like a series of steps. The default step size is 1 and by
changing it to a value of lets say 0.01, we can clearly see the steps of every interval """
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, Regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

""" The DTR has a okay prediction, while the SVR was a poor model comparitively """