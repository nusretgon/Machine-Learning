"""
Created on Fri Aug 12 14:14:18 2022

@author: nusretgon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

df=pd.read_csv("datasets/dataset.csv")
country=df.iloc[:,:1].values
sex=df.iloc[:,4:].values

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ohe=preprocessing.OneHotEncoder()

sex[:,-1] = le.fit_transform(df.iloc[:,-1])
sex = ohe.fit_transform(sex).toarray()
sex=pd.DataFrame(data=sex,index=range(22),columns=("M","W"))
# Because of dummy variable we subtract one column
sex=sex.iloc[:,:1]

country[:,-1]=le.fit_transform(df.iloc[:,:1])
country=ohe.fit_transform(country).toarray()
country=pd.DataFrame(data=country,index=range(22),columns=("FR","TR","US"))

df=df.iloc[:,1:4]
df=pd.concat([country,df],axis=1)
df=pd.concat([df,sex],axis=1)

#  END OIF THE DATA MANIPULATION

""" DEPENDENT AND INDEPENDENT VARIABLES
 We are doing classification so our dependent variable is sex
"""

dependent=sex.values
independent=df.iloc[:,:-1].values

# Split train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(independent,dependent,test_size=0.33)

# Scaling 

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

# Logistic Regression

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
lr_pred=lr.predict(X_test)
cm=confusion_matrix(y_test, lr_pred)
print("Logistic Regression Confusion Matrix")
print(cm)


# KNN Algorithm

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2,metric="minkowski")
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
cm=confusion_matrix(y_test, knn_pred)
print("KNN Confusion Matrix")
print(cm)


# Support Vector Machine

from sklearn.svm import SVC
svm=SVC(kernel="rbf")
svm.fit(X_train,y_train)
svm_pred=svm.predict(X_test)
cm=confusion_matrix(y_test, svm_pred)
print("SVR Confusion Matrix")
print(cm)


# Naive Bayes

from sklearn.naive_bayes import GaussianNB
naive=GaussianNB()
naive.fit(X_train,y_train)
naive_pred=naive.predict(X_test)
cm=confusion_matrix(y_test,naive_pred)
print("Naive Bayes Confusion Matrix")
print(cm)


# Decision Tree 

from sklearn.tree import DecisionTreeClassifier
dTree=DecisionTreeClassifier(criterion="entropy")
dTree.fit(X_train,y_train)
dTree_pred=dTree.predict(X_test)
cm=confusion_matrix(y_test,dTree_pred)
print("Decision Tree Confusion Matrix")
print(cm)


# Random Forest

from sklearn.ensemble import RandomForestClassifier
random=RandomForestClassifier(n_estimators=10,criterion="entropy")
random.fit(X_train,y_train)
random_pred=random.predict(X_test)
cm=confusion_matrix(y_test,random_pred)
print("Random Forest Confusion Matrix")
print(cm)





