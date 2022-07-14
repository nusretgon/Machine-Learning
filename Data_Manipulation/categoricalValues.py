
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

missingDatas=pd.read_csv("../datasets/missingValues.csv")
# process for missing values 
# for numeric values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
age=missingDatas.iloc[:,1:4].values
imputer =imputer.fit(age[:,1:4])
age[:,1:4]=imputer.transform(age[:,1:4])
#--------------------------------------------------------------------------

# We are casting categorical values for calculation
# First version
country=missingDatas.iloc[:,0:1].values
print(country)


from sklearn import preprocessing
# second version
le=preprocessing.LabelEncoder()
country[:,0]=le.fit_transform(missingDatas.iloc[:,0])
print(country)

# last version
ohe=preprocessing.OneHotEncoder()
country=ohe.fit_transform(country).toarray()
print(country)





