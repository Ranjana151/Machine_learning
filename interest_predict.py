# interest rate predication based on credit score of a person
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
from scipy import*

import pandas as pd
from pandas import DataFrame,Series
from pandas.plotting import scatter_matrix
import statsmodels.api as sm


dframe=pd.read_csv("loan.csv.txt")
print(dframe.head())
print(dframe.info())
print(dframe.describe())
print(dframe.tail())
print(dframe['Loan.Length'][0:10])
fico=dframe['FICO.Score']
fico.hist(bins=20)
plt.show()

x=dframe.boxplot('Interest.Rate','FICO.Score',)
x.set_xticklabels(['630','','','','660','','','','690','','','','720','','','','750','','','','780','','','','810','','','','840'])
x.set_xlabel("FICO Score")
x.set_ylabel("Interest Rate in %")
plt.show()


scatter_matrix(dframe,alpha=0.1,color="red",figsize=(8,8),diagonal='hist')
plt.show()
inter_rate=dframe['Interest.Rate']
loan_amount=dframe['Loan.Amount']
fico_score=dframe['FICO.Score']

y=np.array(inter_rate).T
x1=np.array(fico_score).T
x2=np.array(loan_amount).T

x=np.column_stack([x1,x2])
x3=sm.add_constant(x)
#OLS
model=sm.OLS(y,x3)
model_fit=model.fit()
print("the p value are",model_fit.pvalues)
print("the R squared value are",model_fit.rsquared)
