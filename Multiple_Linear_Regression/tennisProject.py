import numpy as np
import pandas as pd

dataset=pd.read_csv("../datasets/tennis.csv")

print(dataset.isnull())
print(dataset.isnull().sum())
# We divide our categoric values.
outlook=dataset.iloc[:,:1].values
windy=dataset.iloc[:,3:4].values
play=dataset[["play"]].values
humidity=dataset[["humidity"]]
# Than we cast them numeric values
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
outlook[:,0]=le.fit_transform(dataset.iloc[:,0])

ohe=preprocessing.OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()
outlook=pd.DataFrame(data=outlook,index=range(14),columns=["overcast","rainy","sunny"])
independent=dataset.iloc[:,1:2]
independent=pd.concat([outlook,independent],axis=1)

windy[:,0]=le.fit_transform(dataset.iloc[:,3:4])
windy=ohe.fit_transform(windy).toarray()
windy=pd.DataFrame(data=windy,index=range(14),columns=["not windy","windy"])
windy=windy[["windy"]]
independent=pd.concat([independent,windy],axis=1)

play[:,0]=le.fit_transform(dataset.iloc[:,4:5])
play=ohe.fit_transform(play).toarray()
play=pd.DataFrame(data=play,index=range(14),columns=["No","Yes"])
play=play[["Yes"]]
independent=pd.concat([independent,play],axis=1)

# End of the data preprocessing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(independent,humidity,train_size=0.66,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_test_predict=lr.predict(x_test)


import statsmodels.api as sm
x=np.append(arr=np.ones((14,1)).astype(int),values=independent,axis=1)

X_l=independent.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(humidity,X_l).fit()
print(model.summary())

# After analyze we should subtract windy column. Windy column affect wrong.

lr.fit(x_train,y_train)
y_test_predict=lr.predict(x_test)
X_l=independent.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(humidity,X_l).fit()
print(model.summary())
