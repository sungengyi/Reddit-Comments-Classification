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
from scipy.stats import mode
from sklearn.model_selection import cross_validate

from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize         

import re


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#import models
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from NaiveBayes import NaiveBayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

num_test_data = 10000 
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
#
#class LemmaTokenizer(object):
#    def __init__(self):
#        self.wnl = WordNetLemmatizer()
#    def __call__(self, articles):
#        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
    
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in re.split('\d|\\\|\s|[,.;:?!]|[/()]|\*',articles)]
    
def averageAcc(cv_results,fold):
    average = 0
    for number in cv_results:
        average+=number
    average /= fold   
    print("Cross-validate",fold,"folds accuracy is:",average)
    return average

 
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
mnb_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))
# 1. 3 multinomial naive bayes: predicting
#------------------------------------------------------------------------------
mnb_predicted = mnb_train_clf.predict(training_data_df['comments'][:num_test_data])
tot_predicted=np.array([mnb_predicted])

# 1. 4 calculate accuracy
#------------------------------------------------------------------------------
print("MNB")
accuracy(mnb_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)
'''
 53.57 with binary = True, 53.85 with False, num_test_data = 30000
1. -----Accuracy: 0.5557 / 10000
2. -----Accuracy: 0.5405666666666666 / 30000
tokenizer=LemmaTokenizer(),
                       strip_accents = 'unicode',
                       stop_words = 'english',
                       lowercase = True,
                       token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                       max_df = 0.5,
                       binary = True)
3. 
-----Execute in 28.745830535888672 sec
-----Accuracy: 0.5407333333333333
tokenizer=LemmaTokenizer(),
                       strip_accents = 'unicode',
                       stop_words = 'english',
                       lowercase = True,
                       token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                       max_df = 0.75,
                       binary = True
4. 
-----Execute in 29.226696014404297 sec
-----Accuracy: 0.5409333333333334
tokenizer=LemmaTokenizer(),
                       strip_accents = 'unicode',
                       stop_words = 'english',
                       lowercase = True,
                       token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
#                       max_df = 0.75,
                       binary = True
5. 
-----Execute in 43.64182090759277 sec
-----Accuracy: 0.5558
tokenizer=LemmaTokenizer(),
                       strip_accents = 'unicode',
                       stop_words = 'english',
                       lowercase = True,
                       token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                       max_df = 0.75,
                       binary = True


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
dct_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))
#

# 2. 3 decision tree: predicting
#------------------------------------------------------------------------------
dct_predicted = dct_train_clf.predict(training_data_df['comments'][:num_test_data])
# 2. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(dct_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)

'''
-----Execute in 143.38194942474365 sec
-----Accuracy: 0.1675
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
lr_predicted = lr_train_clf.predict(training_data_df['comments'][:num_test_data])
tot_predicted=np.append(tot_predicted,[lr_predicted],axis=0)
# 3. 4 calculate accuracy
#------------------------------------------------------------------------------
print("LR")

accuracy(lr_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)

'''
1. num = 30000 binary = True
-----Execute in 31.341885328292847 sec
-----Accuracy: 0.5263666666666666
2.  num = 30000, binary = false
-----Execute in 33.12492775917053 sec
-----Accuracy: 0.5242333333333333
3.num = 30000 binary = True
tokenizer=LemmaTokenizer(),
                       strip_accents = 'unicode',
                       stop_words = 'english',
                       lowercase = True,
                       token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                       max_df = 0.5,
                       min_df = 1,
                       binary = True
-----Execute in 77.83643221855164 sec
-----Accuracy: 0.5247333333333334
'''

#
#
## 4. 1 NB 20 classes
##------------------------------------------------------------------------------
#nb_train_clf = Pipeline([
#        ('vect',CountVectorizer()),
#        ('tfidf',TfidfTransformer()),
#        ('clf', NaiveBayes(20)),
#        ])
## 4. 2 
##------------------------------------------------------------------------------
#start_time = time.time()
#nb_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
#finish_time = time.time()
#print("-----Execute in {} sec".format(finish_time - start_time))
## 4. 3 
##------------------------------------------------------------------------------
#nb_predicted = nb_train_clf.predict(training_data_df['comments'][:num_test_data])
## 4. 4 calculate accuracy
##------------------------------------------------------------------------------
#accuracy(nb_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)
#

# 6.1 SVM
#------------------------------------------------------------------------------
svm_train_clf= Pipeline([
        ('vect',CountVectorizer(binary = True)),
        ('tfidf',TfidfTransformer()),
        ('clf', LinearSVC(C = 0.2)),
        ])
# 6. 2 svm: fitting
#------------------------------------------------------------------------------
start_time = time.time()
svm_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))
# 6. 3 svm: predicting
#------------------------------------------------------------------------------
svm_predicted = svm_train_clf.predict(training_data_df['comments'][:num_test_data])
tot_predicted=np.append(tot_predicted,[svm_predicted],axis=0)
tot_predicted=np.append(tot_predicted,[svm_predicted],axis=0)



# 6. 4 calculate accuracy
#------------------------------------------------------------------------------
print("SVM")

accuracy(svm_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)
'''
1. -----Accuracy: 0.5327666666666667: num 30000
    tokenizer=LemmaTokenizer(),
                       strip_accents = 'unicode',
                       stop_words = 'english',
                       lowercase = True,
                       token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                       max_df = 0.5,
                       min_df = 1,
                       binary = True
2. -----Accuracy: 0.5354333333333333 : num = 30000  
3. -----Accuracy: 0.5544666666666667 : num = 30000 C = 0.2
4. -----Accuracy: 0.5514333333333333 : num = 30000 C = 0.2
    tokenizer=LemmaTokenizer(),
                       strip_accents = 'unicode',
                       stop_words = 'english',
                       lowercase = True,
                       token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                       max_df = 0.5,
                       binary = True
                     
'''

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
        ('clf', KNeighborsClassifier(n_neighbors= 250,
                                     weights='uniform',
                                     algorithm='auto',
                                     n_jobs=-1)),
        ])
    
# 7. 2 k-nearest neighbors: fitting
#------------------------------------------------------------------------------
start_time = time.time()
KN_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))
# 7. 3 k-nearest neighbors: predicting
#------------------------------------------------------------------------------
KN_predicted = KN_train_clf.predict(training_data_df['comments'][:num_test_data])
tot_predicted=np.append(tot_predicted,[KN_predicted],axis=0)
# 7. 4 calculate accuracy
#------------------------------------------------------------------------------
print("KN")

accuracy(KN_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)
'''
3. num 10000
-----Execute in 2.9817392826080322 sec
-----Accuracy: 0.4616
4. num 10000
 -----Accuracy: 0.505
tokenizer=LemmaTokenizer(),
                       strip_accents = 'unicode',
                       stop_words = 'english',
                       lowercase = True,
                       token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                       max_df = 0.5,
                       min_df = 1,
                       binary = True
'''


# 8 1  SGDClassifier
#------------------------------------------------------------------------------
SGD_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', linear_model.SGDClassifier()),
        ])
    
'''
 ------ penalty = "elasticnet"          0.506
                = "l1"                  0.3613
                = "none"                0.5265
                
 ------ loss =  "perceptron"            0.4707
             =  "epsilon_insensitive"   0.5559
'''
# 8. 2   SGDClassifier: fitting
#------------------------------------------------------------------------------
start_time = time.time()
SGD_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))
# 8. 3 SGDClassifier: predicting
#------------------------------------------------------------------------------
SGD_predicted = SGD_train_clf.predict(training_data_df['comments'][:num_test_data])
#tot_predicted=np.append(tot_predicted,[SGD_predicted],axis=0)
# 8. 4 calculate accuracy
#------------------------------------------------------------------------------
print("SGD")
accuracy(SGD_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)




# 9 1  AdaBoostClassifier
#------------------------------------------------------------------------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ADA_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
        learning_rate=5.0, n_estimators=200, random_state=0)),
        ])
# 9. 2  AdaBoostClassifier: fitting
#------------------------------------------------------------------------------
start_time = time.time()
ADA_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))
# 9. 3 SGDClassifier: predicting
#------------------------------------------------------------------------------
ADA_predicted = ADA_train_clf.predict(training_data_df['comments'][:num_test_data])
# 9. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(ADA_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)




# 10. 1  kMeans（Not working, input should be an array)
#------------------------------------------------------------------------------

KM_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', KMeans()),
        ])
# 10. 2   kMeans: fitting
#------------------------------------------------------------------------------
start_time = time.time()
KM_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))

# 10. 3 kMeans: predicting
#------------------------------------------------------------------------------
KM_predicted = KM_train_clf.predict(training_data_df['comments'][:num_test_data])
# 10. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(KM_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)




# 11. 1  DummyClassifier (不好用)
#------------------------------------------------------------------------------

DC_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', DummyClassifier()),
        ])
# 11. 2   DummyClassifier: fitting
#------------------------------------------------------------------------------
start_time = time.time()
DC_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))

# 11. 3 DummyClassifier: predicting
#------------------------------------------------------------------------------
DC_predicted = DC_train_clf.predict(training_data_df['comments'][:num_test_data])
# 11. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(DC_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)



# 12. 1  MLPClassifier(需要调参！！！！)
     
#------------------------------------------------------------------------------
'''
 ----Test with early_stopping = True,learning_rate ="adaptive",max_iter = 100
        #   Runtime:    Accuracy
        #   260s          0.581
        #   158s          0.5772
        #   184s          0.5853
        #   177s          0.5842
        #   306s          0.5794
            101s          0.5717
        
            ----Test withlearning_rate ="adaptive"
            666s          0.5279
            127s          0.5867
            187s          0.5906
            206s          0.577
            116.9s        0.5785
            150s          0.5791
        
        ---Test with solver = "sgd"
        
             Runtime     Accuracy
             52.29s       0.051
             213          0.0614

     
     -----Learning rate and iter time
     - 0.001 -- 216s
     - 1 : ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
-----Execute in 185.87604212760925 sec
    - 0.005 -- 173s -- 1 iter
        -----Accuracy: 0.5734
    - 0.01 -- 208s 
        -----Accuracy: 0.5656
    - 10 -- 178 --acc 0.0505
    - 2 -- 192 --acc 0.0499
    - 0.005 --178 --acc 0.5734
    - 0.01 --208 0.5656
    - 0.008 --176 -acc 0.5655
    - 0.003 --193 --0.5722
    - 0.006 --193 --56.72
    - 0.004 -- 176 0.5752
    - 0.004 iter 2 312  0.5534
    - 0.004 iter 5 717  0.5112
    - 0.004 lemma 
    -----Execute in 152.74802780151367 sec
    -----Accuracy: 0.5785
    

'''
#------------------------------------------------------------------------------

MLP_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', MLPClassifier(learning_rate ="invscaling",
                              learning_rate_init = 0.004,
                              max_iter = 1)), # 122s 0.5651
                                                            # 181s 0.5813
        ])
'''
    #activation : logistic(accuracy too low)
    #             identity(0.5724) 235s
                   tanh   （0.577）  190s
    
    #solver:       lbfgs(not applicable)
                   sgd(too low)
            #activation : logistic(accuracy too low)
            #             identity(0.5724) 235s
                           tanh   （0.577）  190s
            
            #solver:       lbfgs(not applicable)
                           sgd(too low)
'''
# 12. 2   MLPClassifier: fitting
#------------------------------------------------------------------------------
print("----MLP: start executing...")
start_time = time.time()
MLP_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))

# 12. 3 MLPClassifier: predicting
#------------------------------------------------------------------------------
MLP_predicted = MLP_train_clf.predict(training_data_df['comments'][:num_test_data])
# 12. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(MLP_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)






# 13. 1  RandomForestClassifier（太慢了 跑不动）
#------------------------------------------------------------------------------

RF_train_clf = Pipeline([
        ('vect',CountVectorizer(tokenizer=LemmaTokenizer(),
                       strip_accents = 'unicode',
                       stop_words = 'english',
                       lowercase = True,
                       token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                       max_df = 0.5,
                       min_df = 1)),
        ('tfidf',TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=100,n_jobs=3)),
        ])
# 13. 2   RandomForestClassifier: fitting
#------------------------------------------------------------------------------
print("----RandomForest: start executing...")
start_time = time.time()
RF_train_clf.fit(training_data_df['comments'][num_test_data:],training_data_df['subreddit_encoding'][num_test_data:])
finish_time = time.time()
print("-----Execute in {} sec".format(finish_time - start_time))

# 13. 3 RandomForestClassifier: predicting
#------------------------------------------------------------------------------
RF_predicted = RF_train_clf.predict(training_data_df['comments'][:num_test_data])
# 13. 4 calculate accuracy
#------------------------------------------------------------------------------
accuracy(RF_predicted,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)

'''
1. Test with test = 10000
RF_train_clf = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=100,n_jobs=3)),
        ])
-----Execute in 142.33186316490173 sec
-----Accuracy: 0.4342
'''




# -----------------------------------------------------------------------------
#VALIDATE
#------------------------------------------------------------------------------
# Logistic Regression
lr_cv_results = cross_validate(lr_train_clf,training_data_df['comments'],training_data_df['subreddit_encoding'],cv = 7)
sorted(lr_cv_results.keys())
lr_cv_results['fit_time']
lr_cv_results['test_score']
print("LR")
averageAcc(lr_cv_results['test_score'],7)
#Cross-validate 7 folds accuracy is: 0.5447571428571428

#------------------------------------------------------------------------------
# Multiclass Naive Bayes
mnb_cv_results = cross_validate(mnb_train_clf,training_data_df['comments'],training_data_df['subreddit_encoding'],cv = 7)
sorted(mnb_cv_results.keys())
mnb_cv_results['fit_time']
mnb_cv_results['test_score']
print("MNB")
averageAcc(mnb_cv_results['test_score'],7)

#------------------------------------------------------------------------------
# Support Vector Machine
svm_cv_results = cross_validate(svm_train_clf,training_data_df['comments'],training_data_df['subreddit_encoding'],cv = 7)
sorted(svm_cv_results.keys())
svm_cv_results['fit_time']
svm_cv_results['test_score']
print("SVM")
averageAcc(svm_cv_results['test_score'],7)
'''
1. Cross-validate 7 folds accuracy is: 0.5720285714285714: binary = true num 30000
    #'''

#------------------------------------------------------------------------------
# K's Nearest 
kn_cv_results = cross_validate(KN_train_clf,training_data_df['comments'],training_data_df['subreddit_encoding'],cv = 7)
sorted(kn_cv_results.keys())
kn_cv_results['fit_time']
kn_cv_results['test_score']
print("KN")
averageAcc(kn_cv_results['test_score'],7)

#------------------------------------------------------------------------------
# SGD
sgd_cv_results = cross_validate(SGD_train_clf,training_data_df['comments'],training_data_df['subreddit_encoding'],cv = 7)
sorted(sgd_cv_results.keys())
sgd_cv_results['fit_time']
sgd_cv_results['test_score']
print("SGD")
averageAcc(sgd_cv_results['test_score'],7)

#------------------------------------------------------------------------------
# ADA
ada_cv_results = cross_validate(ADA_train_clf,training_data_df['comments'],training_data_df['subreddit_encoding'],cv = 7)
sorted(ada_cv_results.keys())
ada_cv_results['fit_time']
ada_cv_results['test_score']
print("ADA")
averageAcc(ada_cv_results['test_score'],7)


#------------------------------------------------------------------------------
# DCT
dct_cv_results = cross_validate(dct_train_clf,training_data_df['comments'],training_data_df['subreddit_encoding'],cv = 7)
sorted(dct_cv_results.keys())
dct_cv_results['fit_time']
dct_cv_results['test_score']
print("DCT")
averageAcc(dct_cv_results['test_score'],7)



#------------------------------------------------------------------------------
# MLP
mlp_cv_results = cross_validate(MLP_train_clf,training_data_df['comments'],training_data_df['subreddit_encoding'],cv = 7)
sorted(mlp_cv_results.keys())
mlp_cv_results['fit_time']
mlp_cv_results['test_score']
print("MLP")
averageAcc(mlp_cv_results['test_score'],7)
#------------------------------------------------------------------------------
# MLP
rf_cv_results = cross_validate(RF_train_clf,training_data_df['comments'],training_data_df['subreddit_encoding'],cv = 7)
sorted(rf_cv_results.keys())
rf_cv_results['fit_time']
rf_cv_results['test_score']
print("MLP")
averageAcc(rf_cv_results['fit_time'],7)




results = [dct_cv_results, kn_cv_results, lr_cv_results,
           mlp_cv_results, mnb_cv_results,
           rf_cv_results, sgd_cv_results, svm_cv_results]
fit_time = []
score_time = []
score = []
overall = []
label = ["fit_time","score_time","overall","accuracy"]
for element in results:
    fit_time.append(averageAcc(element['fit_time'],7))
    score_time.append(averageAcc(element['score_time'],7))
    overall.append(averageAcc(element['fit_time'],7)+averageAcc(element['score_time'],7))
    score.append(100*averageAcc(element['test_score'],7))

model = ["DT","Kn","LR","MLP","MNB","RF","SGD","SVM"]
fig,ax = plt.subplots()
ax.plot(model,fit_time,label = "fit_time")
ax.plot(model,score_time,label = "score_time")
ax.plot(model,overall,label = "overall")
ax.legend()
plt.show()
 






vp = votepredict(tot_predicted)
print("Vote")

accuracy(vp,training_data_df['subreddit_encoding'][:num_test_data], num_test_data)











