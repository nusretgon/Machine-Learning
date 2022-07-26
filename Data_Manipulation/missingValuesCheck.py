import numpy as np
import pandas as pd

dataset=pd.read_csv("../datasets/tennis.csv")

print(dataset.isnull())
print(dataset.isnull().sum())

dataset1=pd.read_csv("../datasets/missingValues.csv")

print(dataset1.isnull())
print(dataset1.isnull().sum())
