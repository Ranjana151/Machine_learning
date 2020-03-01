import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn import metrics

data=pd.read_csv('tips.csv')
#print(data.head())
#print(data.tail())
#print(data.info())


#Data processing
gender=pd.get_dummies(data['sex'],drop_first=True)
smoke=pd.get_dummies(data['smoker'],drop_first=True)
day=pd.get_dummies(data['day'])
time=pd.get_dummies(data['time'])

final_data=pd.concat([data,gender,smoke,day,time],axis=1)
final_data.rename(columns={'Male':'gender','Yes':'smoker'},inplace=True)
final_data.drop(['sex','smoker','day','time'],axis=1,inplace=True)
print(final_data.tail())

#visualization
sns.heatmap(final_data.corr())
plt.show()


#Data training
x=final_data.drop("tip",axis=1)
y=final_data['tip']

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=40)

model=LinearRegression()
model.fit(X_train,Y_train)

predication=model.predict(X_test)

print(predication[:5])
#Evaluate
mean_square_error=metrics.mean_squared_error(Y_test,predication)
print(mean_square_error)
rms=np.sqrt(mean_square_error)
print(rms)
