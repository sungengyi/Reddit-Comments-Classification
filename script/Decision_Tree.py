# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:38:29 2019

@author: sunge
"""
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from tqdm import tqdm
from numpy import transpose as T
from scipy.stats import stats
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
#encoding subreddits
#------------------------------------------------------------------------------
def encode_subreddit(argument):
    switch = {
           "hockey":0,
            "nba":1,
            "leagueoflegends":2,
            "soccer":3,
            "funny":4,
            "movies":5,
            "anime":6,
            "Overwatch":7,
            "trees":8,
            "GlobalOffensive":9,
            "nfl":10,
            "Askreddit":11,
            "gameofthrones":12,
            "conspiracy":13,
            "worldnews":14,
            "wow":15,
            "europe":16,
            "canada":17,
            "Music":18,
            "baseball":19,             
            }
    return switch.get(argument,20)

start_time = time.time()
#load file
#------------------------------------------------------------------------------
training_data_df = pd.read_csv('../data/encoded_reddit_train.csv')
finish_time = time.time()
print("-----File Loaded in {} sec".format(finish_time - start_time))

#tokenize
#------------------------------------------------------------------------------ 
start_time = time.time()
num_test_data = 60000  
count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(training_data_df['comments'][num_test_data:])
print(train_counts.shape)
finish_time = time.time()
print("-----Tokenized in {} sec".format(finish_time - start_time))


#tf idf
#------------------------------------------------------------------------------
start_time = time.time()
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)
print(train_tfidf.shape)
finish_time = time.time()

print("-----TF*IDF in {} sec".format(finish_time - start_time))

start_time = time.time()
clf = tree.DecisionTreeClassifier()


clf = clf.fit(train_tfidf,training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Fit in {} sec".format(finish_time - start_time))

start_time = time.time()
testing_count = count_vect.transform(training_data_df['comments'][:num_test_data])
testing_tfidf = tfidf_transformer.transform(testing_count)
predicted = clf.predict(testing_tfidf)
finish_time = time.time()

print("-----Predicted in {} sec".format(finish_time - start_time))


#calculate accuracy
#------------------------------------------------------------------------------
accuracy = 0
index = 0
for result in predicted:
    if result == training_data_df['subreddit_encoding'][index]:
          #  print("true",training_data_df['subreddit_encoding'][index])
        #    print("predict",result)
         #   print("------------")
         accuracy+=1
    index+=1
print("-----Accuracy:", accuracy/num_test_data)
    














