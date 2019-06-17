import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

data=pd.read_csv('sms_data.csv',sep='\t',header=None,names=['label','message'])
print(data.head())
#how to map
data['label_number']=data.label.map({'ham':0,'spam':1})
x=data.message
y=data.label_number
print(data.head())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)

vector=CountVectorizer()
vector.fit(x_train)

x_train_dtmatrix=vector.transform(x_train)
print(x_train.shape)
x_test_dtmatrix=vector.transform(x_test)

nb_model=MultinomialNB()
nb_model.fit(x_train_dtmatrix,y_train)

predication=nb_model.predict(x_test_dtmatrix)
print(metrics.accuracy_score(predication,y_test))
