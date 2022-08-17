# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 10:50:45 2022

@author: nusre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/nusre/Desktop/Machine_Learning/datasets/salary2.csv")

x=df.iloc[:,2:4].values

from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=3,init="k-means++")

kmeans.fit(x)
print(kmeans.cluster_centers_)

# WSCC
wssc=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=123)
    kmeans.fit(x)
    wssc.append(kmeans.inertia_)        # inertia gives us WSSC values.


plt.plot(range(1,11),wssc)              # We plot wssc values then we determine number of cluster
