# New Employee salary truth/bluff prediction by HR

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the data
Data = pd.read_csv('Position_Salaries.csv')
X = Data.iloc[:, 1:2].values # To avoid a rank-1 matrix/ array, we mention 1:2 with upper bound not considered
y = Data.iloc[:, 2].values

""" We dont have to split the data into test and train set as the amount of data is less """
# No feature scaling needed as the linear model library has auto feature scaling

""" We are gonna build a Linear Regression Model as well as a Polynomial Regression one and compare the results """
# Fitting the Linear Regression to the data
from sklearn.linear_model import LinearRegression
Lin_Regressor = LinearRegression()
Lin_Regressor.fit(X, y)

""" Unlike the MLR library, the PNL library automatically adds a column of 1's for constant b0 """
# Fitting the Polynomial Regression to the data
# Here we'll call the LinearRegression() again and add polynomial terms on it
from sklearn.preprocessing import PolynomialFeatures

# The degree number is the number of polynomial terms we want to add to X, increase it if the model doesnt fit properly
Poly_Regressor = PolynomialFeatures(degree = 4)
Lin_Regressor_2 = LinearRegression()

X_poly = Poly_Regressor.fit_transform(X)
Lin_Regressor_2.fit(X_poly, y)

# Visualizing the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, Lin_Regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, Lin_Regressor_2.predict(Poly_Regressor.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
Lin_Regressor.predict(6.5)

# Predicting a new result with Polynomial Regression
Lin_Regressor_2.predict(Poly_Regressor.fit_transform(6.5))

""" The new employee was telling the truth, he said his previous salary was 160k and from the data
, Our prediction is 158k, so the employee was telling the truth """ 