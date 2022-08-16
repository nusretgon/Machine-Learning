import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
df=pd.read_csv("../datasets/salary.csv")

x=df.iloc[:,1:2]
y=df.iloc[:,2:]
X=x.values
Y=y.values
# Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,Y)
plt.scatter(X,Y)
plt.plot(X,lr.predict(X))
plt.show()

# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
poly.fit(X,Y)
plt.scatter(X,Y)
plt.plot(X,poly.predict(X))
plt.show()

# Decision Tree 
from sklearn import tree

from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
# n_estimators determine how many decision tree we have.
rf_reg.fit(X,Y)

plt.scatter(X,Y)
plt.plot(X,rf_reg.predict(X))

# R2 SCORE 
print(r2_score(Y,rf_reg.predict(X)))

