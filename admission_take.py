import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data=pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")

#print(data.head())
#print(data.tail())
#print(data.info())
#print(data.describe())

#plt.hist(data['gpa'],bins=35,color="blue")
#plt.xlabel("GPA")
#plt.ylabel("No. of students")
#plt.show()

#plt.hist(data['gre'],bins=30,color="green")
#plt.xlabel("GRE")
#plt.ylabel("No. of students")
#plt.show()

#plt.hist(data['admit'],bins=30,color="red")
#plt.xlabel("admit")
#plt.ylabel("No. of students")
#plt.show()

#sns.jointplot(x="gpa",y="gre",data=data,color="blue",kind="kde")
#plt.show()

#sns.jointplot(x="gpa",y="gre",data=data,color="blue")
#plt.show()
print(data.head())

data=data.fillna(0)


dumpy_rank=pd.get_dummies(data,columns=['rank'])
#print(dumpy_rank.head())
#print(data.tail())
#print(dumpy_rank.tail())

cols_we_need=['admit','gre','gpa']
data=data[cols_we_need].join(dumpy_rank.ix[:,'rank_2':])
print(data.head())
x=data[['gre','gpa','rank_2','rank_3','rank_4']]
y=data['admit']

X_train,X_test,y_train,y_test=train_test_split(x,y ,test_size=0.3,random_state=23)

model=LogisticRegression()
model.fit(X_train,y_train)

x_new=[[640,6.1,0,1,0]]
predication=model.predict(x_new)
print(predication)