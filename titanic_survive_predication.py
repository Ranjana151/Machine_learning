import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn import metrics

train=pd.read_csv("titanic_data.csv")

train=train.drop(columns="Cabin")
print(train.head())
#data visualization

plt.figure(figsize=(5,5))
sns.heatmap(train.corr())
plt.show()
train_new=train.dropna(inplace=True)
print(train.isnull().sum())
# Data manipulation,cleaning data
sex=pd.get_dummies(train['Sex'],drop_first=True)
print(sex.head())

embark=pd.get_dummies(train['Embarked'],drop_first=True)
print(embark.head())

pclass=pd.get_dummies(train['Pclass'],drop_first=True)
print(pclass.head())

train=pd.concat([train,sex,embark,pclass],axis=1)
print(train.head())

train.drop(['Sex','Embarked','Name','PassengerId','Ticket','Pclass'],axis=1,inplace=True)
print(train.head())

X=train.drop("Survived",axis=1)
Y=train["Survived"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=70)
#Model training
model=LogisticRegression()
model.fit(X_train,Y_train)
#Model testing
predication=model.predict(X_test)
print(predication[:5])
print(classification_report(Y_test,predication))
print(accuracy_score(Y_test,predication))

