from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

iris=datasets.load_iris()
x=iris.data
y=iris.target
iris=pd.DataFrame(x,columns=iris.feature_names)
print(iris.head())
print(iris.tail())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=17)
model=LogisticRegression()
model.fit(x_train,y_train)
#try on own datasets
x_new=[[5.4,3.2,1.9,0.3]]
predication=model.predict(x_new)

print(predication)
#print(metrics.accuracy_score(y_test,predication))

