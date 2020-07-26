# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:32:57 2020

@author: Kase
"""

import pandas as pd
import nltk
import re
import numpy as np

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical


import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


Data = pd.read_csv("Tweets.csv")
airline_sentiment=pd.DataFrame(Data["airline_sentiment"]) 
coments =pd.DataFrame(Data["text"])

DataComents = airline_sentiment.join(coments)
Neutral_WE=[]
Positive_WE=[]
Negative_WE=[]
for i in range(len(DataComents)):
    if DataComents["airline_sentiment"][i] == "neutral":
        Neutral_WE.append(DataComents["text"][i])
        #Neutral_sentiment.append(0)
    elif DataComents["airline_sentiment"][i] == "positive":
        Positive_WE.append(DataComents["text"][i])
        #Positive_sentiment.append(1)
    elif DataComents["airline_sentiment"][i] == "negative":
        Negative_WE.append(DataComents["text"][i])
        #Negative_sentiment(-1)        


x_train,x_test,y_train,y_test = [],[],[],[]


ID_counter = 0
for i in Neutral_WE:
    ID_counter +=1
    if ((ID_counter/float(len(Neutral_WE)))*100 <= 80):
        x_train.append(i)
        y_train.append(0)
    else:
        x_test.append(i)
        y_test.append(0)

    

    
ID_counter = 0    


for i in Positive_WE:
    ID_counter +=1
    if ((ID_counter/float(len(Positive_WE)))*100 <= 80):
        x_train.append(i)
        y_train.append(1)
    else:
        x_test.append(i)
        y_test.append(1)

ID_counter = 0    

for i in Negative_WE:
 
    ID_counter +=1
    if ((ID_counter/float(len(Negative_WE)))*100 <= 80):
        x_train.append(i)
        y_train.append(-1)
    else:
        x_test.append(i)
        y_test.append(-1)

y_train =to_categorical(y_train,3)
y_test =to_categorical(y_test,3)

        
from tensorflow.python.keras.preprocessing.text import Tokenizer

tokenizer_obj = Tokenizer()
totalreviews = x_train + x_test
tokenizer_obj.fit_on_texts(totalreviews)

max_length = max([len(s.split()) for s in totalreviews])

vocabSize = len(tokenizer_obj.word_index)+1

x_train_token = tokenizer_obj.texts_to_sequences(x_train)
x_test_token = tokenizer_obj.texts_to_sequences(x_test)

#max_review_length = 300
X_train = sequence.pad_sequences(x_train_token, maxlen=max_length)
X_test = sequence.pad_sequences(x_test_token, maxlen=max_length)


embedding_vecor_length = 160
top_words = 2000
model = Sequential()
model.add(Embedding(vocabSize, embedding_vecor_length, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(3, activation='softmax'))
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=64)
print(model.summary())

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


Review=input("Enter Your Review:")
tokenRev=tokenizer_obj.texts_to_sequences([Review])
revPred = sequence.pad_sequences(tokenRev, maxlen=max_length)
y_pred=model.predict_classes(revPred)
"""
"""


"""    
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
        y_train.append(2)
    else:
        x_test.append(Avg_FV)
        y_test.append(2)
    Avg_FV = np.zeros(300)

"""




"""
from gensim.models import Word2Vec
model_ted =Word2Vec(sentences=Sentences,size=100,window=5,min_count=5,workers=4,sg=0)


tokens = nltk.word_tokenize(string)

from collections import Counter

vocabSize = len(Counter(tokens))


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
    if ((ID_counter/len(Neutral_WE))*100 <= 80):
        x_train.append(Avg_FV)
        y_train.append(0)
    else:
        x_test.append(Avg_FV)
        y_test.append(0)
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
    if ((ID_counter/len(Positive_WE))*100 <= 80):
        x_train.append(Avg_FV)
        y_train.append(1)
    else:
        x_test.append(Avg_FV)
        y_test.append(1)
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
    if ((ID_counter/len(Negative_WE))*100 <= 80):
        x_train.append(Avg_FV)
        y_train.append(2)
    else:
        x_test.append(Avg_FV)
        y_test.append(2)
    Avg_FV = np.zeros(100)
"""















"""
#################################################################
######################DONE WITHOUT WORD2VEC######################

Data = pd.read_csv("Tweets.csv")
airline_sentiment=pd.DataFrame(Data["airline_sentiment"]) 
coments =pd.DataFrame(Data["text"])

DataComents = airline_sentiment.join(coments)
Neutral_WE=[]
Positive_WE=[]
Negative_WE=[]
for i in range(len(DataComents)):
    if DataComents["airline_sentiment"][i] == "neutral":
        Neutral_WE.append(DataComents["text"][i])
        #Neutral_sentiment.append(0)
    elif DataComents["airline_sentiment"][i] == "positive":
        Positive_WE.append(DataComents["text"][i])
        #Positive_sentiment.append(1)
    elif DataComents["airline_sentiment"][i] == "negative":
        Negative_WE.append(DataComents["text"][i])
        #Negative_sentiment(-1)        


x_train,x_test,y_train,y_test = [],[],[],[]


ID_counter = 0
for i in Neutral_WE:
    ID_counter +=1
    if ((ID_counter/float(len(Neutral_WE)))*100 <= 80):
        x_train.append(i)
        y_train.append(0)
    else:
        x_test.append(i)
        y_test.append(0)

    

    
ID_counter = 0    


for i in Positive_WE:
    ID_counter +=1
    if ((ID_counter/float(len(Positive_WE)))*100 <= 80):
        x_train.append(i)
        y_train.append(1)
    else:
        x_test.append(i)
        y_test.append(1)

ID_counter = 0    

for i in Negative_WE:
 
    ID_counter +=1
    if ((ID_counter/float(len(Negative_WE)))*100 <= 80):
        x_train.append(i)
        y_train.append(-1)
    else:
        x_test.append(i)
        y_test.append(-1)

y_train =to_categorical(y_train,3)
y_test =to_categorical(y_test,3)

        
from tensorflow.python.keras.preprocessing.text import Tokenizer

tokenizer_obj = Tokenizer()
totalreviews = x_train + x_test
tokenizer_obj.fit_on_texts(totalreviews)

max_length = max([len(s.split()) for s in totalreviews])

vocabSize = len(tokenizer_obj.word_index)+1

x_train_token = tokenizer_obj.texts_to_sequences(x_train)
x_test_token = tokenizer_obj.texts_to_sequences(x_test)

#max_review_length = 300
X_train = sequence.pad_sequences(x_train_token, maxlen=max_length)
X_test = sequence.pad_sequences(x_test_token, maxlen=max_length)


embedding_vecor_length = 160
top_words = 2000
model = Sequential()
model.add(Embedding(vocabSize, embedding_vecor_length, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(3, activation='softmax'))
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=64)
print(model.summary())

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


Review=input("Enter Your Review:")
tokenRev=tokenizer_obj.texts_to_sequences([Review])
revPred = sequence.pad_sequences(tokenRev, maxlen=max_length)
y_pred=model.predict_classes(revPred)
"""

"""
############################################################
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
"""