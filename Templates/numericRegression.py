import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

veriler = pd.read_csv('../datasets/salary.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,lr.predict(X), color = 'blue')
plt.show()

print('Linear R2 value')
print(r2_score(Y, lr.predict(X)))


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
x_poly = poly.fit_transform(X)
print(x_poly)
lr2 = LinearRegression()
lr2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lr2.predict(poly.fit_transform(X)), color = 'blue')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
x_poly = poly.fit_transform(X)
print(x_poly)
lr2 = LinearRegression()
lr2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lr2.predict(poly.fit_transform(X)), color = 'blue')
plt.show()

print('Polynomial R2 value')
print(r2_score(Y, lr2.predict(poly.fit_transform(X))))


from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

scale_x = sc1.fit_transform(X)

sc2=StandardScaler()
scale_y = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(scale_x,scale_y)

plt.scatter(scale_x,scale_y,color='red')
plt.plot(scale_x,svr_reg.predict(scale_x),color='blue')

plt.show()
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

print('SVR R2 value')
print(r2_score(scale_y, svr_reg.predict(scale_x)))

#Decision Tree Regresyon
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4

plt.scatter(X,Y, color='red')
plt.plot(X,r_dt.predict(X), color='blue')
plt.plot(X,r_dt.predict(Z),color='green')
plt.plot(X,r_dt.predict(K),color='yellow')
plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

print('Decision Tree R2 value')
print(r2_score(Y, r_dt.predict(X)))

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')

plt.plot(X,rf_reg.predict(Z),color='green')
plt.plot(X,r_dt.predict(K),color='yellow')
plt.show()




print('Random Forest R2 value')
print(r2_score(Y, rf_reg.predict(X)))

print(r2_score(Y, rf_reg.predict(K)))
print(r2_score(Y, rf_reg.predict(Z)))

print('-----------------------')
print('Linear R2 value')
print(r2_score(Y, lr.predict(X)))

print('Polynomial R2 value')
print(r2_score(Y, lr2.predict(poly.fit_transform(X))))

print('SVR R2 value')
print(r2_score(scale_y, svr_reg.predict(scale_x)))


print('Decision Tree R2 value')
print(r2_score(Y, r_dt.predict(X)))

print('Random Forest R2 value')
print(r2_score(Y, rf_reg.predict(X)))














