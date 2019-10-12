# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:38:27 2019

@author: Hao Shu
"""


#import models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def __init__(self, x_train, y_train, x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        # Logistic Regression
    def LR(self, c,epoch):
        model = LogisticRegression(C=c,solver='saga',multi_class= 'multinomial',max_iter = epoch)
        model.fit(self.x_train, self.y_train)
        predicts = model.predict(self.x_test)
        #
        scores_LR = cross_val_score(model, self.x_train, self.y_train, cv=7)
        print("Score of LR in Cross Validation", LR_score.mean() * 100)
        print("Logistic Regression accurancy: ", metrics.accuracy_score(self.y_test, predicts))
        
        
        #Decision tree
    def DT(self):
        model = DecisionTreeClassifier()
        model.fit(self.x_train, self.y_train)
        #predicts = model.predict(self.x_test)
        DT_score = cross_val_score(model, self.x_train, self.y_train, cv=7)
        print("Score of decision tree in Cross Validation", DT_score.mean() * 100)
        print("decision tree accuracy: ", metrics.accuracy_score(self.y_test, model.predict(self.x_test)))
        
        
    def svm(self, c):
        model = LinearSVC(C=c)
        model.fit(self.x_train, self.y_train)
        predicts = model.predict(self.x_test)

        SVM_score = cross_val_score(model, self.x_train, self.y_train, cv=7)
        print("Score of SVM in Cross Validation", SVM_score.mean() * 100)
        print("SVM Regression accuracy: ", metrics.accuracy_score(self.y_test, predicts)
        print("Report", classification_report(self.y_test, predicts))

   
