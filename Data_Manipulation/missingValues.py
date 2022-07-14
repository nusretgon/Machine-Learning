
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

missingDatas=pd.read_csv("../datasets/missingValues.csv")

# process for missing values 

# for numeric values
from sklearn.impute import SimpleImputer
# We impute with mean
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")

age=missingDatas.iloc[:,1:4].values

imputer =imputer.fit(age[:,1:4])
age[:,1:4]=imputer.transform(age[:,1:4])


