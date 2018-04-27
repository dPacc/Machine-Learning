# Random Forest Regression
""" Random Forest Regression is an ensemble of Decision Trees, having a lot of Decision Trees
and asking them to vote their opinion and take the average, thats the intuition
behind RFR 
Let's try to implement the Bluff detector that an HR can use to see if the potential new
hire is bluffing about his previos salary. The salary data is given by his previous employer """

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
Data = pd.read_csv('Position_Salaries.csv')
X = Data.iloc[:, 1:2].values
y = Data.iloc[:, 2].values

# Feature Scaling is not needed here
# Fitting the Decision Tree Regression to the dataset
from sklearn.ensemble import RandomForestRegressor

# The n-estimators is the number of trees we want in the RFR
Regressor = RandomForestRegressor(n_estimators = 300)
Regressor.fit(X, y)

# Predicting the new result
y_pred = Regressor.predict(6.5)

# Visualizing the DTR results
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, Regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

""" We have a lot more steps in RFR as there are more splits(more intervals), more trees
doesnt mean more steps, but steps are chosen in a better way 
 Polynomial Regression and RFR are the best models"""