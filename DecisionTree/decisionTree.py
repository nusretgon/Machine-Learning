import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("../datasets/salary.csv")

x=df.iloc[:,1:2]
y=df.iloc[:,2:]

#dataframe to array
X=x.values
Y=y.values

from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)

# Visualize for control

plt.scatter(X,Y,color="red")
plt.plot(X,dt_reg.predict(X),color="black")

print(dt_reg.predict([[11]]))