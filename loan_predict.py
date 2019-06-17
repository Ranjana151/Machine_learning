#predication of good loan or not
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix ,classification_report
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("lending_club_data01.csv.txt")
#print(data.head())
#print(data.tail())
data["good_loans"]=data["bad_loans"].apply(lambda y:'yes' if y==0 else 'no')
print(data.head())
x=data.drop(['bad_loans','good_loans'],axis=1)
y=data['good_loans']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=124)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)

predication=model.predict(x_test)
print(predication)
print(confusion_matrix(y_test,predication))
print(classification_report(y_test,predication))


# By using random forest classifier

rf_model=RandomForestClassifier(n_estimators=150)
rf_model.fit(x_train,y_train)

rf_predication=rf_model.predict(x_test)
print(rf_predication)
print(confusion_matrix(y_test,rf_predication))
print(classification_report(y_test,rf_predication))