# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:26:34 2019

@author: sunge
"""

from sklearn.pipeline import Pipeline
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
from sklearn.linear_model import LogisticRegression

num_test_data = 10000

def accuracy(predicted,true_outcome,num):
    accuracy = 0
    index = 0
    for result in predicted:
        if result == true_outcome[index]:
            accuracy+=1
        index+=1
    print("-----Accuracy:", accuracy/num)
    
    
    
start_time = time.time()
#load file
#------------------------------------------------------------------------------
training_data_df = pd.read_csv('../data/encoded_reddit_train.csv')
finish_time = time.time()
print("-----File Loaded in {} sec".format(finish_time - start_time))



# 2. 1 decision tree
#------------------------------------------------------------------------------
dct_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', tree.DecisionTreeClassifier()),
        ])
# 2. 2 decision tree: fitting
#------------------------------------------------------------------------------
dct_train_clf.fit(training_data_df['comments'],training_data_df['subreddit_encoding'])
# 2. 3 decision tree: predicting
#------------------------------------------------------------------------------
dct_predicted = dct_train_clf.predict(training_data_df['comments'])
# 2. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(dct_predicted,training_data_df['subreddit_encoding'], num_test_data)





# 3. 1 logistic regression
#------------------------------------------------------------------------------
lr_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', LogisticRegression(random_state=0, solver='lbfgs',
                        multi_class='multinomial')),
        ])
# 3. 2 logistic regression: fitting
#------------------------------------------------------------------------------
lr_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
# 3. 3 logistic regression: predicting
#------------------------------------------------------------------------------
lr_predicted = lr_train_clf.predict(training_data_df['comments'][:num_test_data])
# 3. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(lr_predicted,training_data_df['subreddit_encoding'], num_test_data)






















