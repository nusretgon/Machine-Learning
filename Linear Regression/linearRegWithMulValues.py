import pandas as pd

df=pd.read_csv("data.csv")

dependent=df[["SBP"]]
independent=df.iloc[:,:2]

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(df[["Age","Weight"]],df.SBP)
print(lr.coef_)         # m1 and m2
print(lr.intercept_)    # b
prediction=lr.predict([[18,70]])