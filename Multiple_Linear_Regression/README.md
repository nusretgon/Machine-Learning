* In multiple linear regression, we have intercept and coefficients.
* Coefficients are basically weight of independent variable.
* If coefficient is high this variable affects dependent variable much.

![Ekran Alıntısı](https://user-images.githubusercontent.com/72438433/179427981-4a083e23-0431-4775-898f-58ad26df7aa0.PNG)

* In linearRegWithMulValues.py we get these values.

![s](https://user-images.githubusercontent.com/72438433/179428114-234eb3f4-8c6a-41d9-be91-1f1ec6a5c844.PNG)

### Second Project
* In second project we use statsmodels.api and we get coefficient and p values.
* Our p value approximately should be p<Significance Value(0,05).
![First model](https://user-images.githubusercontent.com/72438433/180601099-115b396b-404c-44f0-a748-20bb214c76ea.PNG)

* x5 = 0,717. So we need to eliminate x5.
![Ekran Alıntısı](https://user-images.githubusercontent.com/72438433/180601186-941218a5-103c-419d-83cf-75ad4953d1d3.PNG)

* Of course we eliminate x5 but now x6 turn x5. Now we don't have higher value than SL(Signifance Level).
* If you want you can eliminate more variable but it may not always works.

### Tennis Project
* In tennisProject.py we have multiple independent values.
* You can download and try with dataset(in datasets folder).
* Dataset is not big so our results are not correct enough.
* Results; 

![as](https://user-images.githubusercontent.com/72438433/181013670-ea73514c-fc42-4795-9b7b-76bd4193013d.PNG)

* And we get this statsmodel summary;

![Düzeltme](https://user-images.githubusercontent.com/72438433/181494664-cbae2ea9-72b6-4a19-b8d5-501e114c7f2c.PNG)

* As you see we get one variable high p value from 0.5. So we should subtract this variable.
* We use backward elimination.

![Ekran Alıntısı](https://user-images.githubusercontent.com/72438433/181714785-e4dc8948-0237-40fd-a4f8-437a92ccf184.PNG)

* BEFORE AFTER CORRECT VALUES for tennis project;

![beforeElemination](https://user-images.githubusercontent.com/72438433/181906370-0e53c2c2-07d0-43b0-a359-8339f71b8f49.PNG)
