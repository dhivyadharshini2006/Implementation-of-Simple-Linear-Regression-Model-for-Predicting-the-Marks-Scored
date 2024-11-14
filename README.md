# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Dhivya Dharshini B
RegisterNumber:212223240031  
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:


![image](https://github.com/user-attachments/assets/6182a7e7-af97-4035-b10a-c88d3dce3052)

![image](https://github.com/user-attachments/assets/96845ce2-a3d3-47c6-921c-27f191e8aad1)

![image](https://github.com/user-attachments/assets/d8cb806c-e9e3-4163-8141-2ad05571b7ba)

![image](https://github.com/user-attachments/assets/93dc9902-52a2-4604-90a6-2c195e4a65c5)


![Screenshot 2024-08-29 114303](https://github.com/user-attachments/assets/3fe4d593-4b2a-4cc7-80d9-696ce3da86b5)

![image](https://github.com/user-attachments/assets/de4f8774-a598-44aa-a75a-1bc66257eb40)


![image](https://github.com/user-attachments/assets/7f2934a4-718b-423d-8cc7-22942fcf6a94)

![image](https://github.com/user-attachments/assets/0cbbce4d-4e4b-4a16-aa2b-ce9e987748ec)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
