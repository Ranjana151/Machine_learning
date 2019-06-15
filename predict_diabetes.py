import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
col=['pregnant','glucose','bp','skin','insulin_level','bmi','pedigree','age','diabetes_level']
data=pd.read_csv('pima.csv.txt',names=col)
print(data.head())
feature=['pregnant','insulin_level','bmi','age']
x=data[feature]
y=data.diabetes_level

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=20)
model=LogisticRegression()
model.fit(x_train,y_train)

predication=model.predict(x_test)
print(predication[1])
print(metrics.accuracy_score(y_test,predication))