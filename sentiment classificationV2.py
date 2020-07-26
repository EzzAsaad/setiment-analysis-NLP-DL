# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:32:57 2020

@author: Kase
"""

import pandas as pd
import nltk
import re
import numpy as np


Data = pd.read_csv("Tweets.csv")
airline_sentiment=pd.DataFrame(Data["airline_sentiment"]) 
coments =pd.DataFrame(Data["text"])

DataComents = airline_sentiment.join(coments)
Neutral_WE=[]
Positive_WE=[]
Negative_WE=[]
for i in range(len(DataComents)):
    if DataComents["airline_sentiment"][i] == "neutral":
        Neutral_WE.append(nltk.word_tokenize(re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', " ", DataComents["text"][i])))
        #Neutral_sentiment.append(0)
    elif DataComents["airline_sentiment"][i] == "positive":
        Positive_WE.append(nltk.word_tokenize(re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', " ", DataComents["text"][i])))
        #Positive_sentiment.append(1)
    elif DataComents["airline_sentiment"][i] == "negative":
        Negative_WE.append(nltk.word_tokenize(re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', " ", DataComents["text"][i])))
        #Negative_sentiment(-1)        


x_train,x_test,y_train,y_test = [],[],[],[]
import spacy
nlp = spacy.load('en_core_web_md')

Avg_FV = np.zeros(300)
ID_counter = 0
for i in Neutral_WE:
    doc = nlp(" ".join(i))
    for index in range(len(i)):
         Avg_FV += doc[index].vector
         
    Avg_FV /=len(i)
    ID_counter +=1
    if ((ID_counter/float(len(Neutral_WE)))*100 <= 80):
        x_train.append(Avg_FV)
        y_train.append(0)
    else:
        x_test.append(Avg_FV)
        y_test.append(0)
    Avg_FV = np.zeros(300)
    

    
ID_counter = 0    
Avg_FV = np.zeros(300)

for i in Positive_WE:
    doc = nlp(" ".join(i))
    for index in range(len(i)):
         Avg_FV += doc[index].vector
    
    Avg_FV /=len(i) 
    ID_counter +=1
    if ((ID_counter/float(len(Positive_WE)))*100 <= 80):
        x_train.append(Avg_FV)
        y_train.append(1)
    else:
        x_test.append(Avg_FV)
        y_test.append(1)
    Avg_FV = np.zeros(300)

Avg_FV = np.zeros(300)
ID_counter = 0    

for i in Negative_WE:
    doc = nlp(" ".join(i))
    for index in range(len(i)):
         Avg_FV += doc[index].vector
    
    Avg_FV /=len(i) 
    ID_counter +=1
    if ((ID_counter/float(len(Negative_WE)))*100 <= 80):
        x_train.append(Avg_FV)
        y_train.append(-1)
    else:
        x_test.append(Avg_FV)
        y_test.append(-1)
    Avg_FV = np.zeros(300)
    
from sklearn import svm
svc = svm.SVC(kernel ='linear', C = 4).fit(x_train, y_train)

y_predict = svc.predict(x_test)

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_predict)*100,"%")

Review=input("Enter Your Review:")

Review_Test = nltk.word_tokenize(re.sub(r"[\W]", " ",Review))
doc = nlp(" ".join(Review_Test))
for index in range(len(Review_Test)):
     Avg_FV += doc[index].vector
    
Avg_FV /=len(i) 
print(svc.predict([Avg_FV]))