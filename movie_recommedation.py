from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pandas import DataFrame,Series

def get_title_from_index(index):
    return df[df.index==index]["title"].values[0]



def get_index_from_title(title):
    return df[df.title==title]["index"].values[0]



df=pd.read_csv("movie_dataset.csv")
#print(df.head())
#print(df.columns)
features=['keywords','cast','genres','director']
for feature in features:
    df[feature]=df[feature].fillna("")

def combine_feature(row):
    try:
        return row['keywords'] +" "+row['cast']+" "+row['genres']+" "+row['director']
    except:
        print("Error:",row)

df["combined_features"]=df.apply(combine_feature,axis=1)
#print("Combined Features:",df["combined_features"].head())

cv=CountVectorizer()
count_matrix=cv.fit_transform((df["combined_features"]))

cosine_sim=cosine_similarity(count_matrix)
movie_user_like="Avatar"

movie_index=get_index_from_title(movie_user_like)
similar_movies=list(enumerate(cosine_sim[movie_index]))

sorted_smilar_movies=sorted(similar_movies,key= lambda x:x[1],reverse=True)
i=0
for element in sorted_smilar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>20:
        break