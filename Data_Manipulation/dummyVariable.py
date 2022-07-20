import pandas as pd
import numpy as np

dataset1=pd.read_csv("datasets/dataset.csv")

country=dataset1.iloc[:,:1].values
sex=dataset1.iloc[:,-1:].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
sex[:,-1] = le.fit_transform(dataset1.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
sex = ohe.fit_transform(sex).toarray()
# We are naming columns
print(list(range(22)))
sex = pd.DataFrame(data=sex, index = range(22), columns = ['e','k'])
sex=sex.iloc[:,:1]

country[:,-1] = le.fit_transform(dataset1.iloc[:,:1])
country = ohe.fit_transform(country).toarray()

country=pd.DataFrame(data=country,index=range(22),columns=["fr","tr","us"])

dataset=dataset1.copy()
dataset=dataset.iloc[:,1:4]

dataset=pd.concat([country,dataset],axis=1)
dataset=pd.concat([dataset,sex],axis=1)
