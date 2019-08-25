import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

suv_data=pd.read_csv("suv_data.csv")
print(suv_data.head())
gender=pd.get_dummies(suv_data['Gender'],drop_first=True)
final_data=pd.concat([suv_data,gender],axis=1)
print(final_data.head())
X=suv_data.iloc[:,2:4].values
print(X)
Y=suv_data.iloc[:,4]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=50)

model=LogisticRegression()
model.fit(X_train,Y_train)
X_new=[[19,50000]]
predication=model.predict(X_new)
print(predication)