import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("../datasets/salary.csv")
#print(df.isnull().sum())

x=df.iloc[:,1:2]
y=df.iloc[:,2:]
X=x.values
Y=y.values

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,Y)

plt.scatter(X,Y,color="red")
plt.plot(x,lr.predict(X),color="blue")

# polynomial regression
# 2.nd degree
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(X)
lr2=LinearRegression()
lr2.fit(x_poly,y)


plt.scatter(X,Y,color="red")
plt.plot(X,lr2.predict(poly.fit_transform(X)),color="orange")
plt.show()
# Polynomial Regression 4.th degree
poly=PolynomialFeatures(degree=4)
x_poly1=poly.fit_transform(X)
lr2=LinearRegression()
lr2.fit(x_poly1,y)


plt.scatter(X,Y,color="red")
plt.plot(x,lr.predict(X),color="blue")
plt.plot(X,lr2.predict(poly.fit_transform(X)),color="orange")
plt.show()

print("Linear: ",lr.predict([[11]]))
print("Polynomial:",lr2.predict(poly.fit_transform([[11]])))