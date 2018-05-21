""" The dataset contains data of 50 startups that a VC Firm wants to invest in
and we have to decide which of them would be the best investment """

# Let's solve this using Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

""" Since we have categorical data in one of the IV columns, we need to Encode it """
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEnco = LabelEncoder()
X[:, 3] = LabelEnco.fit_transform(X[:, 3])
OneHotEnco = OneHotEncoder(categorical_features = [3])
X = OneHotEnco.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap we remove one of the dummy variables
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

# No Feature Scaling needed as the Linear Regression library takes care of it

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = Regressor.predict(X_test)

""" Since the number of IV's and DV totally are 5, we need a 5D to plot, which is not that easy
So, we are going to use 'Backward Elimination' method to omit the IV's which doesnt have
a significant impact on the DV """
# We need to use another library called the 'statsmodels' for this
# The IV with the highest Probability/P-value(whichever is greater than the SL = 0.05) will be manually eliminated
import statsmodels.formula.api as sm

# The sm library will not take the 'b0' into account
# To avoid this we need to multiply a column of ones assigned to x0 and the append it to the X matrix
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]

# To find the p-values, we'll use the OLS(Ordinary Least Square) method into a new Regressor
# We'll call the summary() funtion on this Regressor to get this info
Regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
Regressor_OLS.summary()

# The 3rd IV(x2) has the highest p-value, so eliminate it
X_optimal = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# The 2nd IV(x1) now has the highest p-value
X_optimal = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# The 3rd IV(x2) has the highest p-value
X_optimal = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# The 3rd IV(x2) has a p-value > SL = 0.05, so eliminate that as well
X_optimal = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

""" We can now say that the IV with the highest impact in the Profit is the R&D Spending"""