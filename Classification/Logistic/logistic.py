# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 16:30:52 2022

@author: nusre
"""

import pandas as pd

df=pd.read_csv("datasets/dataset.csv")
df=df.iloc[:,1:]
independent=df.iloc[:,:-1].values
dependent=df.iloc[:,-1:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(independent,dependent,train_size=0.66)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
 
X_test=sc.fit_transform(x_test)
X_train=sc.fit_transform(x_train)

from sklearn.linear_model import LogisticRegression

logR=LogisticRegression(random_state=0)
logR.fit(X_train,y_train)
y_pred=logR.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

