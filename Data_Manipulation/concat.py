import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("../datasets/dataset.csv")

# We are casting categorical values for calculation
# First version
country=dataset.iloc[:,0:1].values
print(country)

from sklearn import preprocessing
# second version
le=preprocessing.LabelEncoder()
country[:,0]=le.fit_transform(dataset.iloc[:,0])
df = pd.DataFrame(country)
# last version
ohe=preprocessing.OneHotEncoder()
country=ohe.fit_transform(country).toarray()

dataset=dataset.iloc[:,1:]

country=pd.DataFrame(data=country)

lastDataset=pd.concat([country,dataset])