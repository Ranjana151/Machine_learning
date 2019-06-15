import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as  plt
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import LinearRegression
from sklearn import metrics

house=pd.read_csv('home_data.csv.txt')
#print(house.head())
#print((house.tail()))
#print(house.info())
#print(house.describe())
#print(house.columns)
#plt.scatter(house.sqft_living,house.price)
#plt.xlabel('sqrt of house')
#plt.ylabel('house of the price')
#plt.show()

#sns.heatmap(house.corr())
#plt.show()

#sns.distplot(house['price'],color='red')
#plt.show()
#sns.boxplot(x='zipcode',y='price',data=house)
#plt.show()
X=house[[ 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'condition','grade','long','sqft_living15']]
y=house['price']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=7)

model=LinearRegression()
model.fit(x_train,y_train)

predication=model.predict(x_test)
print(house.head())
house1=house[house['id']==6414100192 ]
print(house1)
print(predication[1])
print(model.coef_)
pddataframe_=pd.DataFrame(model.coef_,X.columns,columns=['Coefficient Value'])
print(pddataframe_)
#print(model.intercept_())
mean_square_error=metrics.mean_squared_error(y_test,predication)
print(mean_square_error)
rms=np.sqrt(mean_square_error)
print(rms)
print(X.shape)


