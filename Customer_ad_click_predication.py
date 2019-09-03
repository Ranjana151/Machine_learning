import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

data=pd.read_csv("advertising.csv")
print(data.head())
#print(data.info())
print(data.columns)

#Data visualization
sns.heatmap(data.corr())
plt.show()

sns.lmplot(x='Age',y='Clicked on Ad',data=data)
plt.show()

#Data processing
data=data.drop(['Ad Topic Line','City','Country'],axis=1)
data['Timestamp']=pd.to_datetime(data['Timestamp'])
data['Month']=data['Timestamp'].dt.month
data['Day of the month']=data['Timestamp'].dt.month
data['Day of the week']=data['Timestamp'].dt.dayofweek
data['Hour']=data['Timestamp'].dt.hour
data=data.drop(['Timestamp'],axis=1)

print(data.head())


#Model training using Linear model
x=data[['Daily Time Spent on Site','Age', 'Area Income','Daily Internet Usage',
       'Male','Month','Day of the month','Day of the week']]
y=data['Clicked on Ad']

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=20)

model=LogisticRegression()
model.fit(X_train,Y_train)

predication=model.predict(X_test)
print(predication[:5])

#Model evaluation

print(metrics.confusion_matrix(Y_test,predication))
print(metrics.accuracy_score(Y_test,predication))

#Model training using Decision Tree
model1=DecisionTreeClassifier()
model1.fit(X_train,Y_train)
predication=model1.predict(X_test)
print(predication[:5])

#Decision Tree model evaluation
print(metrics.accuracy_score(Y_test,predication))
print(metrics.confusion_matrix(Y_test,predication))
