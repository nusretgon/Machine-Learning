# Librarys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Reading process
dataset=pd.read_csv("../datasets/sales.csv")

# We divide columns to process.
months=dataset[["Months"]]
sales=dataset[["Sales"]]

# Dividing train and test
from sklearn.model_selection import train_test_split
# First independent variable second dependent variable
x_train,x_test,y_train,y_test=train_test_split(months,sales,train_size=0.66,random_state=0)

# We scale values
"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)
"""
# Linear Regression process
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
# We learned y train from x train
lr.fit(x_train,y_train)
# We predict values from x_test
# prediction is y_test
prediction=lr.predict(x_test)

# We must sort because if we dont plots will be weird
x_train=x_train.sort_index()
y_train=y_train.sort_index()


# Visualization
plt.plot(x_train,y_train)
plt.title("Sales")
plt.xlabel("Months")
plt.ylabel("Money")
plt.plot(x_test,lr.predict(x_test))
