from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data=pd.read_csv('wineQualityReds.csv')
#print(data.head())
print(data.info())
#print(data.columns)
#print(data.iloc[:,0])

data=data.rename(columns={'quality':'Wine_rating'})
print(data.tail())
print(data.iloc[:,12])

x=data.drop('Wine_rating',axis=1)
y=data['Wine_rating']

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2)

model=LinearRegression()
model.fit(X_train,Y_train)

predication=model.predict(X_test)

print(predication[:5])
