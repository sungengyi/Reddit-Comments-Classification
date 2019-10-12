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

#import models
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from NaiveBayes import NaiveBayes

num_test_data = 9000

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
training_data_df = pd.read_csv(r'../data/encoded_reddit_train.csv')
finish_time = time.time()
print("-----File Loaded in {} sec".format(finish_time - start_time))



# 1. 1 multinomial naive bayes
#------------------------------------------------------------------------------
mnb_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', MultinomialNB()),
        ])
# 1. 2 multinomial naive bayes: fitting
#------------------------------------------------------------------------------
start_time = time.time()
mnb_train_clf.fit(training_data_df['comments'],training_data_df['subreddit_encoding'])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))
# 1. 3 multinomial naive bayes: predicting
#------------------------------------------------------------------------------
mnb_predicted = mnb_train_clf.predict(training_data_df['comments'][:num_test_data])
# 1. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(mnb_predicted,training_data_df['subreddit_encoding'], num_test_data)




# 2. 1 decision tree
#------------------------------------------------------------------------------
dct_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', tree.DecisionTreeRegressor()),
        ])
# 2. 2 decision tree: fitting
#------------------------------------------------------------------------------
start_time = time.time()
dct_train_clf.fit(training_data_df['comments'],training_data_df['subreddit_encoding'])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))
#

# 2. 3 decision tree: predicting
#------------------------------------------------------------------------------
dct_predicted = dct_train_clf.predict(training_data_df['comments'][:num_test_data])
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
start_time = time.time()
lr_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))
# 3. 3 logistic regression: predicting
#------------------------------------------------------------------------------
lr_predicted = lr_train_clf.predict(training_data_df['comments'][:num_test_data])
# 3. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(lr_predicted,training_data_df['subreddit_encoding'], num_test_data)




# 4. 1 
#------------------------------------------------------------------------------
nb_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', NaiveBayes(20)),
        ])
# 3. 2 logistic regression: fitting
#------------------------------------------------------------------------------
start_time = time.time()
nb_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))# 3. 3
#logistic regression: predicting
#------------------------------------------------------------------------------
nb_predicted = nb_train_clf.predict(training_data_df['comments'][:num_test_data])
# 3. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(nb_predicted,training_data_df['subreddit_encoding'], num_test_data)


# 5. 1 multinomial naive bayes
#------------------------------------------------------------------------------
mnb_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', MultinomialNB()),
        ])
# 5. 2 multinomial naive bayes: fitting
#------------------------------------------------------------------------------
start_time = time.time()
mnb_train_clf.fit(training_data_df['comments'],training_data_df['subreddit_encoding'])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))
# 5. 3 multinomial naive bayes: predicting
#------------------------------------------------------------------------------
mnb_predicted = mnb_train_clf.predict(training_data_df['comments'])
# 5. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(mnb_predicted,training_data_df['subreddit_encoding'], num_test_data)


# 6.1 SVM
#------------------------------------------------------------------------------
svm_train_clf= Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', LinearSVC(1.0)),
        ])
# 6. 2 svm: fitting
#------------------------------------------------------------------------------
svm_train_clf.fit(training_data_df['comments'],training_data_df['subreddit_encoding'])
# 6. 3 svm: predicting
#------------------------------------------------------------------------------
svm_predicted = svm_train_clf.predict(training_data_df['comments'])
# 6. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(svm_predicted,training_data_df['subreddit_encoding'], num_test_data)



















