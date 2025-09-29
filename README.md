# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries and Load Dataset.
2. Preprocess the Data.
3. Split the Dataset.
4. Train the Decision Tree Classifier.
5. Make Predictions and Evaluate the Model.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Reena K
RegisterNumber:  212224040272
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data.head()
data["salary"]=le.fit_transform(data["salary"])
data
x=data[["satisfaction_level","last_evaluation","number_project","time_spend_company"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
import pandas as pd
dt.predict(pd.DataFrame([[0.5,0.8,9,2]], columns=x.columns))

```
## Output:

<img width="306" height="36" alt="Screenshot 2025-09-29 093945" src="https://github.com/user-attachments/assets/ad374c49-de8c-49cf-b494-59c09f6953c4" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
