# **If we need to manipulate datasets, we need these code blocks.**
### Missing Values
* I recently upload new file missingValueCheck.py. If we want to check any missing value. We can use isnull() function.

![missingValue](https://user-images.githubusercontent.com/72438433/181493028-185e8c9c-8d75-4da5-add9-9f3b13a3226f.PNG)
* First of all, I will show how to deal with missing values.
* In missingValues.py  we will change missing values with mean of datas.
* Results are ;

![Untitled](https://user-images.githubusercontent.com/72438433/179108641-c6bffcc7-6784-4197-9634-490fc19b5196.png)





### Categorical Values
* We should cast categoric values to numeric because we can't use in calculation categoric values.
* All steps shown in picture ;

![Untitled1](https://user-images.githubusercontent.com/72438433/179117433-3940f8b7-297d-4e99-a06f-30f76d6a3597.png)
* After first project, we get 0 1 2 for country. But we can't say US better than TR or FR.
* So we did second process.

### We rarely use scaling process.
* We have lots of data and we want these datas similar.
* So we can use scaling process.

![Ekran Alıntısı](https://user-images.githubusercontent.com/72438433/179349566-49fe5b29-b1df-4299-9f5e-ad65fb12352a.PNG)

### Dummy Variable

* Sometimes we have more than one variable to represent one feature. For example,sex.
* In dummyVariable.py firstly we cast sex column to numeric values. Then we get Mens and Womens.
* Men mean not women. Women mean not men. So we don't need any second column.
* If we use just men column, we solve problem.

![Adsız](https://user-images.githubusercontent.com/72438433/179936770-cba3cbe3-7aff-4e49-b7b8-06c2c8d31d98.png)

