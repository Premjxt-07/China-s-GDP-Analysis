""" 

    The following program uses non-linear regression to analyze china's GDP over the years 1960-2014.
    The dataset has two columns the first one contains the year while the 2nd contains the annual GDP in US $.


"""
# DEPENDENCIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# IMPORTING DATASET

df = pd.read_csv("china_gdp.csv")
# print(df.head(10))

x_data, y_data = (df["Year"].values, df["Value"].values)

# VISUALIZING THE DATASET TO CHOOSE THE BEST REGRESSION
""" 
plt.figure(figsize=(8, 5))
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show() """

# The graph closely resembles to that of an logistic function . Hence it would be best to use logistic regression
# since the growth is slow in the beginning then there is growth in the middle and then it dies.

""" 
            The formula for logistic regression is :    y = 1 / 1 + e^(a(X-b)) where
            a controls the curves steepness
            b glides the graph over the x axis

"""


def sigmoid(x, beta_1, beta_2):
    y = 1/(1+np.exp(-beta_1*(x-beta_2)))
    return y


beta_1 = 0.1
beta_2 = 1990

# EXAMPLE OF A SAMPLE SIGMOID FUNCTION`

""" y_pred = sigmoid(x_data,beta_1,beta_2)
plt.plot(x_data, y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')
plt.show() """

# NORMALIZATION OF DATA

xdata = x_data/max(x_data)
ydata = y_data/max(y_data)


""" 
    we can use __curve_fit__ which uses non-linear least squares to fit our sigmoid function, to data.
    Optimal values for the parameters so that the sum of the squared residuals of sigmoid(xdata, *popt) - ydata is minimized.

"""
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
# print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))  The best parameters are 

# TO VISUALIZE THE PLOT 

""" x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show() """

msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# MODEL TRAIN USING THE BEST FIT PARAMETERS
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# PREDICITION OF TEST SET
y_predict = sigmoid(test_x, *popt)

# ACCURACY
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_predict - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_predict - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_predict , test_y) )

# y_predict = sigmoid([2020], *popt)
# print(y_predict * max(y_data))


