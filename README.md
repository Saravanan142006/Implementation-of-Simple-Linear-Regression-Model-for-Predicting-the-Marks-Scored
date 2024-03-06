# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Saravanan M
RegisterNumber: 212223080050
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
#displaying X
X
Y=df.iloc[:,1].values
#displaying Y
Y


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying Y pred
Y_pred

#displaying actual values
Y_test

#graph plotting area
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()  
*/
```

## Output:
![pic2](https://github.com/Saravanan142006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161738800/43a32692-c9be-430d-8190-0ba900491a90)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
