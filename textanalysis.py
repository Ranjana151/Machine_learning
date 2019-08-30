import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame

train_text=['go home tonight','go take a cap','soccer! go play soccer....','are you mad']

vector=CountVectorizer()
vector.fit(train_text)

print(vector.get_feature_names())

train_text_dtmatrix=vector.transform(train_text)

print(train_text_dtmatrix.toarray())

text=pd.DataFrame(train_text_dtmatrix.toarray(),columns=vector.get_feature_names())
print(text.head())


