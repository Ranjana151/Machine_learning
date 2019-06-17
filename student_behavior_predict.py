#student clustering based on following features
#Alchol problem and has a value 0-6
#deviant behaviour
#Violent behaviour scale
#depression
#parental activity
#parental presence
#family connectedness
#selfesteem

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
data=pd.read_csv("health_data.csv.txt")
#print(data.head())
data.columns=map(str.upper,data.columns)
data=data.dropna()
print(data.head())
print(data.tail())
print(data.info())

cluster=data[['ALCEVR1','MAREVER1','ALCPROBS1','DEVIANT1','VIOL1','DEP1','ESTEEM1','SCHCONN1','PARACTV','PARPRES','FAMCONCT']]

cluster_s=cluster.copy()
cluster_s['ALCEVR1']=preprocessing.scale(cluster_s['ALCEVR1'].astype('float64'))
cluster_s['MAREVER1']=preprocessing.scale(cluster_s['MAREVER1'].astype('float64'))
cluster_s['ALCPROBS1']=preprocessing.scale(cluster_s['ALCPROBS1'].astype('float64'))
cluster_s['DEVIANT1']=preprocessing.scale(cluster_s['DEVIANT1'].astype('float64'))
cluster_s['VIOL1']=preprocessing.scale(cluster_s['VIOL1'].astype('float64'))
cluster_s['DEP1']=preprocessing.scale(cluster_s['DEP1'].astype('float64'))
cluster_s['ESTEEM1']=preprocessing.scale(cluster_s['ESTEEM1'].astype('float64'))
cluster_s['SCHCONN1']=preprocessing.scale(cluster_s['SCHCONN1'].astype('float64'))
cluster_s['PARACTV']=preprocessing.scale(cluster_s['PARACTV'].astype('float64'))
cluster_s['PARPRES']=preprocessing.scale(cluster_s['PARPRES'].astype('float64'))
cluster_s['FAMCONCT']=preprocessing.scale(cluster_s['FAMCONCT'].astype('float64'))

cluster_train,cluster_test=train_test_split(cluster_s,test_size=0.3,random_state=222)

clusters=range(1,11)
mean_dist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(cluster_train)
    mean_dist.append(sum(np.min(cdist(cluster_train,model.cluster_centers_,'euclidean'),axis=1))/cluster_train.shape[0])

plt.plot(clusters,mean_dist)
plt.xlabel('Number of cluster')
plt.ylabel('Ave.Distance')
plt.title('Elbow method for our k values')
plt.show()

model=KMeans(n_clusters=3)
model.fit(cluster_train)
pca_2=PCA(2)
plot_columns=pca_2.fit_transform(cluster_train)
plt.scatter(x=plot_columns[:,0],y=plot_columns[:,1],c=model.labels_)
plt.xlabel('Canonical Var 1')
plt.ylabel('Canonical Var 2')
plt.title('Scatter plot for 3 clusters')
plt.show()