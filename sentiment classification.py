# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:00:07 2020

@author: Kase
"""

import pandas as pd
import nltk
import re
import numpy as np


Data = pd.read_csv("Tweets.csv")
Id = pd.DataFrame(Data["tweet_id"])
airline_sentiment=pd.DataFrame(Data["airline_sentiment"]) 
coments =pd.DataFrame(Data["text"])

DataComents = Id.join(airline_sentiment.join(coments))
Sentences = []
Neutral_WE=[]
Positive_WE=[]
Negative_WE=[]
for i in range(len(DataComents)):
    if DataComents["airline_sentiment"][i] == "neutral":
        Neutral_WE.append(nltk.word_tokenize(re.sub(r"[\W]", " ", DataComents["text"][i])))
        #Neutral_sentiment.append(0)
    elif DataComents["airline_sentiment"][i] == "positive":
        Positive_WE.append(nltk.word_tokenize(re.sub(r"[\W]", " ", DataComents["text"][i])))
        #Positive_sentiment.append(1)
    else:
        Negative_WE.append(nltk.word_tokenize(re.sub(r"[\W]", " ", DataComents["text"][i])))
        #Negative_sentiment(-1)
    Sentences.append(nltk.word_tokenize(re.sub(r"[\W]", " ", DataComents["text"][i])))
        
from gensim.models import Word2Vec
model_ted =Word2Vec(sentences=Sentences,size=100,window=5,min_count=5,workers=4,sg=0)
Neutral = {"id":0,"airline_sentiment":0,"AVGFV":[]}
Positive = {"id":0,"airline_sentiment":1,"AVGFV":[]}
Negative = {"id":0,"airline_sentiment":-1,"AVGFV":[]}

Data_neutral = []
Data_positive = []
Data_negative = []

x_train,x_test,y_train,y_test = [],[],[],[]

Avg_FV = np.zeros(100)
ID_counter = 0
for i in Neutral_WE:
    for j in i:
        try:
            Avg_FV += model_ted.wv[j]
        except:
            Avg_FV += np.zeros(100)
    
    Avg_FV /=len(i)
    ID_counter +=1
    Neutral = {"id":ID_counter,"airline_sentiment":0,"AVGFV":Avg_FV}
    Data_neutral.append(Neutral)
    if ((ID_counter/len(Neutral_WE))*100 <= 80):
        x_train.append(Avg_FV)
        y_train.append(0)
    else:
        x_test.append(Avg_FV)
        y_test.append(0)
    Neutral = {"id":0,"airline_sentiment":0,"AVGFV":[]}
    Avg_FV = np.zeros(100)
    

    
ID_counter = 0    
Avg_FV = np.zeros(100)

for i in Positive_WE:
    for j in i:
        try:
            Avg_FV += model_ted.wv[j]
        except:
            Avg_FV += np.zeros(100)
    
    Avg_FV /=len(i) 
    ID_counter +=1
    Positive = {"id":ID_counter,"airline_sentiment":1,"AVGFV":Avg_FV}
    Data_positive.append(Positive)
    if ((ID_counter/len(Positive_WE))*100 <= 80):
        x_train.append(Avg_FV)
        y_train.append(1)
    else:
        x_test.append(Avg_FV)
        y_test.append(1)
    Positive = {"id":0,"airline_sentiment":1,"AVGFV":[]}
    Avg_FV = np.zeros(100)

Avg_FV = np.zeros(100)
ID_counter = 0    

for i in Negative_WE:
    for j in i:
        try:
            Avg_FV += model_ted.wv[j]
        except:
            Avg_FV += np.zeros(100)
    
    Avg_FV /=len(i) 
    ID_counter +=1
    Negative = {"id":ID_counter,"airline_sentiment":-1,"AVGFV":Avg_FV}
    Data_negative.append(Negative)
    if ((ID_counter/len(Negative_WE))*100 <= 80):
        x_train.append(Avg_FV)
        y_train.append(-1)
    else:
        x_test.append(Avg_FV)
        y_test.append(-1)
    Negative = {"id":0,"airline_sentiment":-1,"AVGFV":[]}
    Avg_FV = np.zeros(100)
    
from sklearn import svm
svc = svm.SVC(kernel ='linear', C = 4).fit(x_train, y_train)

y_predict = svc.predict(x_test)

from sklearn import metrics

print("accuracy:",round(metrics.accuracy_score(y_test, y_predict)*100,1))

Review=input("Enter Your Review:")

Review_Test = [nltk.word_tokenize(re.sub(r"[\W]", " ",Review))]
for i in Review_Test:
    try:
        Avg_FV += model_ted.wv[j]
    except:
        Avg_FV += np.zeros(100)
    
Avg_FV /=len(i) 
print(svc.predict([Avg_FV]))
            
"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

y_predict = knn.predict(x_test)
print("accuracy:",metrics.accuracy_score(y_test, y_predict)*100)
"""

