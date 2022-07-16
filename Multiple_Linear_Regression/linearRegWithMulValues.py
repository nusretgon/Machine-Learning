import pandas as pd
import math
df=pd.read_csv("../datasets/data.csv")

dependent=df[["SBP"]]
independent=df.iloc[:,:2]

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(df[["Age","Weight"]],df.SBP)
print("\nCoefficients:",lr.coef_,"\n")         # m1 and m2
print("Intercept:",lr.intercept_,"\n")          # b
prediction=lr.predict([[18,70]])

print("Age:18, Weigth:70 SBP:",math.floor(prediction[0]))