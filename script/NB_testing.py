import time
import numpy as np
import NaiveBayes as nb

from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB



def binarize(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j]>0:
                X[i,j]=1
            else:
                X[i,j]=0

def test():

    nb_samples = 2000
    nb_rounds = 10
    x = np.zeros((nb_rounds))
    y = np.zeros((nb_rounds))
    x_time = 0.0
    y_time = 0.0
    for i in range(nb_rounds):
        bnbdata_X, bnbdata_Y = make_classification(n_samples=nb_samples, 
                                                   n_features=20, n_informative=20,
                                                   n_classes=5, n_redundant=0)
        binarize(bnbdata_X)
        
        
        
        bnb = MultinomialNB()
        start_time = time.time()
        y_pred_official = bnb.fit(bnbdata_X,bnbdata_Y).predict(bnbdata_X)
        finish_time = time.time()
        y_time += (finish_time-start_time)
        
        
        
        mnb = nb.NaiveBayes(num_class=20)
        start_time = time.time()
        mnb.fit(bnbdata_X,bnbdata_Y)
        y_pred_scratch = mnb.predict(bnbdata_X)
        finish_time = time.time()
        x_time += (finish_time-start_time)
        
        print("mnb: ",(bnbdata_Y != y_pred_scratch).sum(),"bnb: ",(bnbdata_Y != y_pred_official).sum())
        y[i] =(bnbdata_Y != y_pred_official).sum()
        x[i] =(bnbdata_Y != y_pred_scratch).sum()
    print("mnb_ave_time: ",x_time/nb_rounds,"bnb_avg_time: ", y_time/nb_rounds)
    return np.var(x),np.var(y),np.average(x),np.average(y)

print(test())