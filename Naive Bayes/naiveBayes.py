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


# Naive Bayes

from sklearn.naive_bayes import GaussianNB
naive=GaussianNB()
naive.fit(X_train,y_train)
naive_pred=naive.predict(X_test)
cm=confusion_matrix(y_test,naive_pred)
print("Naive Bayes Confusion Matrix")
print(cm)
