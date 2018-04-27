# Support Vector Regression
""" We are going to solve the same problem of HR predicting if the employee is 
bluffing or telling the truth about his previous salary using SVR"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
Data = pd.read_csv('Position_Salaries.csv')
X = Data.iloc[:, 1:2].values
y = Data.iloc[:, 2].values

""" SVR class(less common model) doesn't have auto feature scaling"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fittng the model to the dataset
from sklearn.svm import SVR
""" One of the params of SVR is kernel, we can choose between linear, polynomial or rbf """
Regressor = SVR(kernel = 'rbf')
Regressor.fit(X, y)

# Predicting the new result
y_pred = sc_y.inverse_transform(Regressor.predict(sc_X.fit_transform(np.array[[6.5]])))

# Visualizing the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, Regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff(SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

""" Less commonly used model """"