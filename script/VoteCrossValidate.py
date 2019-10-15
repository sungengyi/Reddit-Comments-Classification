# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:19:57 2019

@author: sunge
"""

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
from sklearn.model_selection import KFold


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#import models
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from NaiveBayes import NaiveBayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier


def accuracy(predicted,true_outcome,num,index):
    accuracy = 0
    for result in predicted:
        if result == true_outcome[index]:
            
            accuracy+=1
        index+=1
    print("-----Accuracy:", accuracy/num)
    return accuracy/num

def votepredict(tot_predicted):
    tot_predicted = np.transpose(tot_predicted)
    vote_predicted = [mode(w).mode[0] for w in tot_predicted]
    return vote_predicted


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
    
#def averageAcc(cv_results,fold):
#    average = 0
#    for number in cv_results:
#        average+=number
#    average /= fold   
#    print("Cross-validate",fold,"folds accuracy is:",average)

 
start_time = time.time()
#load file
#------------------------------------------------------------------------------
training_data_df = pd.read_csv(r'../data/encoded_reddit_train.csv')
finish_time = time.time()
print("-----File Loaded in {} sec".format(finish_time - start_time))

#
#kf = KFold(n_splits = 7)
#X=training_data_df['comments']
#y = training_data_df['comments']
#kf.get_n_splits(X)
#for train_index, test_index in kf.split(X):
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]



def VoteAndCrossValidate(X,Y,splits,num_data):
        final_acc = 0
        kf = KFold(n_splits = splits)
        kf.get_n_splits(X)
        index = 0
        iter_num = 1
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
     
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
            mnb_train_clf.fit(X_train,y_train)
            finish_time = time.time()
            print("-----Execute in {} sec".format(finish_time - start_time))
            # 1. 3 multinomial naive bayes: predicting
            #------------------------------------------------------------------------------
            mnb_predicted = mnb_train_clf.predict(X_test)
            tot_predicted=np.array([mnb_predicted])       
            # 1. 4 calculate accuracy
            #------------------------------------------------------------------------------
            print(iter_num,"MNB")
            accuracy(mnb_predicted,y_test,num_data,index)
                   
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
            lr_train_clf.fit(X_train,y_train)
            finish_time = time.time()
            print("-----Execute in {} sec".format(finish_time - start_time))
            # 3. 3 logistic regression: predicting
            #------------------------------------------------------------------------------
            lr_predicted = lr_train_clf.predict(X_test)
            tot_predicted=np.append(tot_predicted,[lr_predicted],axis=0)
            # 3. 4 calculate accuracy
            #------------------------------------------------------------------------------
            print(iter_num,"LR")      
            accuracy(lr_predicted,y_test,num_data,index)
          
            
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
            svm_train_clf.fit(X_train,y_train)
            finish_time = time.time()
            print("-----Execute in {} sec".format(finish_time - start_time))
            # 6. 3 svm: predicting
            #------------------------------------------------------------------------------
            svm_predicted = svm_train_clf.predict(X_test)
            tot_predicted=np.append(tot_predicted,[svm_predicted],axis=0)
            tot_predicted=np.append(tot_predicted,[svm_predicted],axis=0)
            #when there is no weight for svm, accuracy is 0.5727
            #when there is weight, accuracy is 0.5799
            # 6. 4 calculate accuracy
            #------------------------------------------------------------------------------
            print(iter_num,"SVM")
            
            accuracy(svm_predicted,y_test,num_data,index)
           
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
            KN_train_clf.fit(X_train,y_train)
            finish_time = time.time()
            print("-----Execute in {} sec".format(finish_time - start_time))
            # 7. 3 k-nearest neighbors: predicting
            #------------------------------------------------------------------------------
            KN_predicted = KN_train_clf.predict(X_test)
            tot_predicted=np.append(tot_predicted,[KN_predicted],axis=0)
            # 7. 4 calculate accuracy
            #------------------------------------------------------------------------------
            print(iter_num,"KN")
            
            accuracy(KN_predicted,y_test,num_data,index)
            
            
            # 8 1  SGDClassifier
            #------------------------------------------------------------------------------
            SGD_train_clf = Pipeline([
                    ('vect',CountVectorizer()),
                    ('tfidf',TfidfTransformer()),
                    ('clf', linear_model.SGDClassifier()),
                    ])
            # 8. 2   SGDClassifier: fitting
            #------------------------------------------------------------------------------
            start_time = time.time()
            SGD_train_clf.fit(X_train,y_train)
            finish_time = time.time()
            print("-----Execute in {} sec".format(finish_time - start_time))
            # 8. 3 SGDClassifier: predicting
            #------------------------------------------------------------------------------
            SGD_predicted = SGD_train_clf.predict(X_test)
            tot_predicted=np.append(tot_predicted,[SGD_predicted],axis=0)
            # 8. 4 calculate accuracy
            #------------------------------------------------------------------------------
            print(iter_num,"SGD")
            accuracy(SGD_predicted,y_test,num_data,index)
            
                        
            # 12. 1  MLPClassifier(需要调参！！！！)
            #------------------------------------------------------------------------------
            
            MLP_train_clf = Pipeline([
                    ('vect',CountVectorizer()),
                    ('tfidf',TfidfTransformer()),
                    ('clf', MLPClassifier(early_stopping = True,learning_rate ="adaptive",max_iter = 100)),
                    ])
            # 12. 2   MLPClassifier: fitting
            #------------------------------------------------------------------------------
            print("MLP: Start executing...")
            start_time = time.time()
            MLP_train_clf.fit(X_train,y_train)
            finish_time = time.time()
            print("-----Execute in {} sec".format(finish_time - start_time))
            
            # 12. 3 MLPClassifier: predicting
            #------------------------------------------------------------------------------
            MLP_predicted = MLP_train_clf.predict(X_test)
            # 12. 4 calculate accuracy
            #------------------------------------------------------------------------------
            print(iter_num,"MLP")
            accuracy(MLP_predicted,y_test,num_data,index)
            
            

        
            
            vp = votepredict(tot_predicted)
            print(iter_num,"Vote")
            
            final_acc +=accuracy(vp,y_test,num_data,index)
            index+=num_data
            iter_num+=1
        print("Final acc:",final_acc / splits)  
        return final_acc / splits


start_time = time.time()
start = 0
end = 70000
split = 7
X = training_data_df['comments'][:end]
y = training_data_df['subreddit_encoding'][:end]
acc = VoteAndCrossValidate(X,y,split,(int)(end - start)/split)


finish_time = time.time()
print("-----File Loaded in {} sec".format(finish_time - start_time))









