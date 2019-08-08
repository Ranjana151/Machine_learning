import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import metrics
import joblib

df=pd.read_csv("dataset.csv")
print(df.head())

x=df.drop(['Label'],axis=1)
y=df["Label"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

model=RandomForestClassifier(n_estimators=100,max_depth=5)
model.fit(x_train,y_train)
joblib.dump(model,"rf_malaria_100_5")

predictions=model.predict(x_test)
print(predictions[:4])
print(metrics.classification_report(predictions,y_test))
