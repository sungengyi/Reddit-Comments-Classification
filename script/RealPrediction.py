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
import string
from tqdm import tqdm
from numpy import transpose as T
from sklearn import tree
from scipy.stats import mode


from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#import models
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
    
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
        ('vect',CountVectorizer(tokenizer=LemmaTokenizer(),
                       strip_accents = 'unicode',
                       stop_words = 'english',
                       lowercase = True,
                       token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                       max_df = 0.8,
                       binary = True)),
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
tot_predicted=np.array([mnb_predicted])
# 1. 4 calculate accuracy
#------------------------------------------------------------------------------

'''
 53.57 with binary = True, 53.85 with False, num_test_data = 30000
'''


# 3. 1 logistic regression
#------------------------------------------------------------------------------
lr_train_clf = Pipeline([
        ('vect',CountVectorizer(tokenizer=LemmaTokenizer(),
                       strip_accents = 'unicode',
                       stop_words = 'english',
                       lowercase = True,
                       token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                       max_df = 0.5,
                       min_df = 1,
                       binary = True)),
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
lr_predicted = lr_train_clf.predict(test_data_df['comments'])
tot_predicted=np.append(tot_predicted,[lr_predicted],axis=0)
# 3. 4 calculate accuracy
#------------------------------------------------------------------------------


'''
1. num = 30000 binary = True
-----Execute in 31.341885328292847 sec
-----Accuracy: 0.5263666666666666
2.  num = 30000, binary = false
-----Execute in 33.12492775917053 sec
-----Accuracy: 0.5242333333333333
'''


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
tot_predicted=np.append(tot_predicted,[svm_predicted],axis=0)
# 6. 4 calculate accuracy
#------------------------------------------------------------------------------



# 7. 1 k-nearest neighbors
#------------------------------------------------------------------------------
KN_train_clf = Pipeline([
        ('vect',CountVectorizer(tokenizer=LemmaTokenizer(),
                       strip_accents = 'unicode',
                       stop_words = 'english',
                       lowercase = True,
                       token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                       max_df = 0.4,
                       binary = True)),
        ('tfidf',TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors= 250)),
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
tot_predicted=np.append(tot_predicted,[KN_predicted],axis=0)
# 7. 4 calculate accuracy
#------------------------------------------------------------------------------



# 12. 1  MLPClassifier(需要调参！！！！)
#------------------------------------------------------------------------------

MLP_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', MLPClassifier(learning_rate ="adaptive")),
        ])
# 12. 2   MLPClassifier: fitting
#------------------------------------------------------------------------------
print("-----MLP: Start executing..."）
start_time = time.time()
MLP_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))

# 12. 3 MLPClassifier: predicting
#------------------------------------------------------------------------------
MLP_predicted = MLP_train_clf.predict(training_data_df['comments'][:num_test_data])
tot_predicted=np.append(tot_predicted,[MLP_predicted],axis=0)
# 12. 4 calculate accuracy
#------------------------------------------------------------------------------






# Final step
#------------------------------------------------------------------------------
vp = votepredict(tot_predicted)
vp = transback(vp)
df = pd.DataFrame({'Category': vp})
df.to_csv(r'solution.csv')














