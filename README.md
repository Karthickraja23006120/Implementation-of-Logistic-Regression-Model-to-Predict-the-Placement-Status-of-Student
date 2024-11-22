# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Karthick Raja K
RegisterNumber:  212223240066
*/

import pandas as pd
df = pd.read_csv("/content/Placement_Data.csv")
print(df.head())
df1 = df.copy()
df1 = df1.drop(['sl_no', 'salary'], axis = 1)
# print(df1.head())
print(df1.isnull().sum())
print(df1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1['gender'] = le.fit_transform(df1['gender'])
df1['ssc_b'] = le.fit_transform(df1['ssc_b'])
df1['ssc_p'] = le.fit_transform(df1['ssc_p'])
df1['hsc_b'] = le.fit_transform(df1['hsc_b'])
df1['hsc_p'] = le.fit_transform(df1['hsc_p'])
df1['hsc_s'] = le.fit_transform(df1['hsc_s'])
df1['degree_t'] = le.fit_transform(df1['degree_t'])
df1['workex'] = le.fit_transform(df1['workex'])
df1['status'] = le.fit_transform(df1['status'])
df1["specialisation"]=le.fit_transform(df1["specialisation"])

print(df1)

x = df1.iloc[:,:-1]
# print(x)

y = df1['status']
# print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)

print(lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]))
```

## Output:

![Screenshot 2024-09-21 100624](https://github.com/user-attachments/assets/44b4bedd-8d38-41b1-a691-779968283540)
![Screenshot 2024-09-21 100635](https://github.com/user-attachments/assets/536c2459-14f7-4ec9-9a1a-4a59719c0fc2)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
