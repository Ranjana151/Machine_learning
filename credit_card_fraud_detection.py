#Detection of fraud Credit cards

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report ,accuracy_score
import pandas as pd

data=pd.read_csv('creditcard.csv')
#print(data.shape)
#print(data.head())
#print(data.info())
data=data.sample(frac=0.1,random_state=1)
print(data.shape)
#data.hist(figsize=(25,25))
#plt.show()
fraud=data[data['Class']==1]
Valid=data[data['Class']==0]



outline_fraction=len(fraud)/float(len(Valid))
print(outline_fraction)
print("Fraud cases:()",format(len(fraud)))
print("Valid cases:()",format(len(Valid)))

#Visualization

correl=data.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(correl,vmax=.8,square=True)
plt.show()

columns=data.columns.tolist()
columns=[c for c in columns if c not in ['Class']]
target="Class"

X=data[columns]
Y=data[target]
#define a random_state
state=1

#define a classifier
classifiers={
    "Isolation forest":IsolationForest(
        max_samples=len(X),
        contamination=outline_fraction,
        random_state=state),
    "Local Outlier Fraction":LocalOutlierFactor(
        n_neighbors=20,
        contamination=outline_fraction

    )
}

#fit the model
n_outlier=len(fraud)

for i,(clf_name,clf) in enumerate(classifiers.items()):
    if clf_name=="Local Outlier Fraction":
        y_predict=clf.fit_predict(X)
        scores_pred=clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_predict=clf.predict(X)

# Reshape the data
    y_predict[y_predict==1]=0
    y_predict[y_predict==-1]=1
    n_erros=(y_predict!=Y).sum()

    print('{}:{}'.format(clf_name,n_erros))
    print(accuracy_score(Y,y_predict))
    print(classification_report(Y,y_predict))