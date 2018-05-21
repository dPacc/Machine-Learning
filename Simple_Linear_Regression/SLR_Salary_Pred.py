# Simple Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset using pandas
Data = pd.read_csv('Salary_Data.csv')
X = Data.iloc[:, :1].values
y = Data.iloc[:, 1].values

# Splitting the data to train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

# Feature Scaling (We dont need Feature Scaling for SLR as the library will take care of it)
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.fit_transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

# Applying the regressor to the train set
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = Regressor.predict(X_test)

# Visualizing training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, Regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, Regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

