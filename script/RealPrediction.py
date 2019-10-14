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
from scipy.stats import mode
from sklearn.model_selection import cross_validate


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#import models
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from NaiveBayes import NaiveBayes

num_test_data = 30000

def accuracy(predicted,true_outcome,num):
    accuracy = 0
    index = 0
    for result in predicted:
        if result == true_outcome[index]:
            accuracy+=1
        index+=1
    print("-----Accuracy:", accuracy/num)

def votepredict(tot_predicted):
    tot_predicted = np.transpose(tot_predicted)
    vote_predicted = [mode(w).mode[0] for w in tot_predicted]
    return vote_predicted

def transback(pred):
    subreddits = pd.read_csv(r'../data/subreddits.csv')
    word = [subreddits['0'][i] for i in pred]
    return word

    
start_time = time.time()
#load file
#------------------------------------------------------------------------------
training_data_df = pd.read_csv(r'../data/encoded_reddit_train.csv')
test_data_df = pd.read_csv(r'../data/original_data/reddit_test.csv')
finish_time = time.time()
print("-----File Loaded in {} sec".format(finish_time - start_time))



# 1. 1 multinomial naive bayes
#------------------------------------------------------------------------------
mnb_train_clf = Pipeline([
        ('vect',CountVectorizer(binary = True)),
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
mnb_predicted = mnb_train_clf.predict(test_data_df['comments'])
#tot_predicted=np.array([mnb_predicted])
# 1. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(mnb_predicted,training_data_df['subreddit_encoding'], num_test_data)
'''
 53.57 with binary = True, 53.85 with False, num_test_data = 30000
'''


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
        ('vect',CountVectorizer(binary = True)),
        ('tfidf',TfidfTransformer()),
        ('clf', LogisticRegression(random_state=0, solver='lbfgs',
                        multi_class='multinomial', max_iter = 300)),])
# 3. 2 logistic regression: fitting
#------------------------------------------------------------------------------
start_time = time.time()
lr_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))
# 3. 3 logistic regression: predicting
#------------------------------------------------------------------------------
lr_predicted = lr_train_clf.predict(training_data_df['comments'][:num_test_data])
#tot_predicted=np.append(tot_predicted,[lr_predicted],axis=0)
# 3. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(lr_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)

'''
1. num = 30000 binary = True
-----Execute in 31.341885328292847 sec
-----Accuracy: 0.5263666666666666
2.  num = 30000, binary = false
-----Execute in 33.12492775917053 sec
-----Accuracy: 0.5242333333333333
'''



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
        ('vect',CountVectorizer(binary = True)),
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
mnb_predicted = mnb_train_clf.predict(test_data_df['comments'])
tot_predicted=np.append(tot_predicted,[mnb_predicted],axis=0)
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
svm_predicted = svm_train_clf.predict(test_data_df['comments'])

# 6. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(svm_predicted,training_data_df['subreddit_encoding'], num_test_data)


# 7. 1 k-nearest neighbors
#------------------------------------------------------------------------------
KN_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=5)),
        ])
# 7. 2 k-nearest neighbors: fitting
#------------------------------------------------------------------------------
start_time = time.time()
KN_train_clf.fit(training_data_df['comments'],training_data_df['subreddit_encoding'])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))
# 7. 3 k-nearest neighbors: predicting
#------------------------------------------------------------------------------
KN_predicted = KN_train_clf.predict(training_data_df['comments'])
tot_predicted=np.append(tot_predicted,[mnb_predicted],axis=0)
# 7. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(mnb_predicted,training_data_df['subreddit_encoding'], num_test_data)


# Final step
#------------------------------------------------------------------------------
vp = votepredict(tot_predicted)
vp = transback(vp)
df = pd.DataFrame({'Category': vp})


# -----------------------------------------------------------------------------
#VALIDATE
#------------------------------------------------------------------------------

lr_cv_results = cross_validate(lr_train_clf,training_data_df['comments'],training_data_df['subreddit_encoding'],cv = 7)
sorted(lr_cv_results.keys())
lr_cv_results['fit_time']
lr_cv_results['test_score']

mnb_cv_results = cross_validate(mnb_train_clf,training_data_df['comments'],training_data_df['subreddit_encoding'],cv = 7)
sorted(mnb_cv_results.keys())
mnb_cv_results['fit_time']
mnb_cv_results['test_score']

svm_cv_results = cross_validate(svm_train_clf,training_data_df['comments'],training_data_df['subreddit_encoding'],cv = 7)
sorted(svm_cv_results.keys())
svm_cv_results['fit_time']
svm_cv_results['test_score']













