import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.svm import SVC
from pandas import DataFrame as df
# prediction of  gender on the basis of voice
data=pd.read_csv("voice.csv")
print(data.head())
#print(data.info())
gender=pd.get_dummies(data['label'],drop_first=True)
#print(gender.tail())

data=pd.concat([data,gender],axis=1)
print(data.head())
data.drop(['label'],axis=1,inplace=True)
print(data.head())
data.rename(columns={'male':'gender'},inplace=True)
print(data.columns)



#Visualization

sns.heatmap(data.corr())
plt.show()

#training
x=data.drop("gender",axis=1)
y=data['gender']
#print(x)
#print(y)
X_tain,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=50)

svc=SVC()
svc.fit(X_tain,Y_train)

#predication
predication=svc.predict(X_test)
print(predication[:3])

