# Polynomial Regression (pr)
# author Zifeng Li
# last update 2024.2.19 BeiJing time

## this code will use polynomial model to make regressions 
## set windows to determine "windows" days' data input
## set pre_w to determine the next "pre_w" days' data to view if the model is useful for unknown data

### if you want to make ture predictions, MA lines and volums should be provided 

import numpy as np
from math import ceil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def pr_main(xdata,ydata,power):

    xdata = np.power(xdata,power)
    # regression model close = b + a1*time + a2*time**2 + ... = [1,time,time**2,...,time**n]@[b,a1,a2,...,an]
    X_train,X_test,Y_train,Y_test = train_test_split(xdata, ydata, test_size=0.3, random_state=5)
    poly_features = PolynomialFeatures(degree=1, include_bias=False)
    
    MV_train =  np.concatenate(X_train, axis=1)
    X_poly_train = poly_features.fit_transform(MV_train)
    Reg = LinearRegression()
    Reg.fit(X_poly_train, Y_train)

    # prediction
    MV_test =  np.concatenate(X_test, axis=1)
    X_poly_test = poly_features.fit_transform(MV_test)
    Y_pred = Reg.predict(X_poly_test)

    # error
    mse = mean_squared_error(Y_test, Y_pred)
    print(f'Mean Squared Error: {mse}')

    # visualization
    print('Intercept:', Reg.intercept_)
    print('Coefficients:', Reg.coef_)
    
    MV_fit =  np.concatenate(xdata, axis=1)
    x_poly_fit = poly_features.fit_transform(MV_fit)
    y_fit = Reg.predict(x_poly_fit)

    plt.figure(1);plt.title("Curv fitting")
    plt.plot(xdata,y_fit)
    plt.xlabel("date")
    plt.legend()

    plt.show() # show figrues


## start
pr_main()

