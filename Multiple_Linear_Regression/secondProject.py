import pandas as pd
import numpy as np

dataset1=pd.read_csv("../datasets/dataset.csv")
country=dataset1.iloc[:,:1].values
sex=dataset1.iloc[:,-1:].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
sex[:,-1] = le.fit_transform(dataset1.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
sex = ohe.fit_transform(sex).toarray()

# We are naming columns
sex = pd.DataFrame(data=sex, index = range(22), columns = ['M','W'])
sex=sex.iloc[:,:1]
country[:,-1] = le.fit_transform(dataset1.iloc[:,:1])
country = ohe.fit_transform(country).toarray()
country=pd.DataFrame(data=country,index=range(22),columns=["fr","tr","us"])
dataset=dataset1.copy()

dataset=dataset.iloc[:,2:4]
dataset=pd.concat([country,dataset],axis=1)
dataset=pd.concat([dataset,sex],axis=1)
height=dataset1[["height"]]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dataset,height,train_size=0.66,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

# Backward elemination
import statsmodels.api as sm
x=np.append(arr=np.ones((22,1)).astype(int),values=dataset,axis=1)
X_l=dataset.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(height,X_l).fit()
print(model.summary())

# From here we eleminate variable have the highest p value
X_l=dataset.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(height,X_l).fit()
print(model.summary())
