import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("../datasets/salary.csv")
#print(df.isnull().sum())

x=df.iloc[:,1:2]
y=df.iloc[:,2:]
X=x.values
Y=y.values

# SVR (Support Vector Regression)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_sc=sc.fit_transform(X)
sc1=StandardScaler()
Y_sc=sc1.fit_transform(Y)

from sklearn.svm import SVR

svr_reg=SVR(kernel="rbf")
svr_reg.fit(X_sc,Y_sc)
plt.scatter(X_sc,Y_sc,color="red")
plt.plot(X_sc,svr_reg.predict(X_sc),color="blue")

print(svr_reg.predict([[11]]))