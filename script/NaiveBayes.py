import numpy as np
import math
from sklearn.utils import check_X_y

class NaiveBayes:
    def __init__(self, num_class = 2, ls = True):
        self.ls = ls
        self.num_class = num_class
        
        
    def fit(self, X_features, Y_quality):
        X_features, Y_quality = check_X_y(X_features, Y_quality, accept_sparse='csr', dtype='float', order="C",
                                          accept_large_sparse='warn' != 'liblinear')
        self.X_features = X_features
        self.Y_outcomes = Y_quality
        self.n,self.m = self.X_features.shape
        self.N = np.zeros((self.num_class),dtype=int)
#        for i in range(self.Y_outcomes.shape[0]):
#            self.N[self.Y_outcomes[i]]+=1
        for i in range(self.num_class):
            self.N[i] = len([w for w in self.Y_outcomes if w==i])
        
        self.w = np.zeros([self.num_class,self.m],dtype=float)
        self.theta = np.ones_like(self.w,dtype = float)
        self.ww = np.ones_like(self.w,dtype = float)
        self.logp = np.ones_like(self.w,dtype = float)
        # make array w that stores all features p(xi|y=c)

        for i in range(self.n):
            self.w[self.Y_outcomes[i]] += self.X_features[i]

                

        if self.ls == True :
            self.laplace_smooth()
        else:
#            for j in range(self.m):
#                for i in range(self.num_class):
#                    self.theta[i,j]=self.w[i,j]/self.N[i]
            for i in range(self.num_class):
                self.theta[i] = self.w[i]/self.N[i]
        if self.num_class == 2:
            for j in range(self.m):
                self.ww[0,j] = math.log((1-self.theta[1,j])/(1-self.theta[0,j]))
                self.ww[1,j] = math.log((self.theta[1,j])/(self.theta[0,j]))
        else:
            pass
                
                
 
               
    def laplace_smooth(self):
            for i in range(self.num_class):
                self.theta[i] = (self.w[i]+1)/(self.N[i]+2)

        
    def predict(self, dataset):
        if self.num_class == 2:
            
            XtW = np.zeros_like(dataset)
            for i in range(dataset.shape[0]):
                for j in range(self.m):
                    XtW[i,j] =(self.ww[1,j]-self.ww[0,j])*dataset[i,j]
            decision = np.zeros([dataset.shape[0]], dtype = int)
            for i in range(dataset.shape[0]):
                result = math.log(self.N[1]/self.N[0])+ self.ww[0].sum() + XtW[i].sum()
                if result > 0:
                    decision[i]=1
                else:
                    decision[i]=0
            return decision
        
        else:
            decision = np.zeros([dataset.shape[0]], dtype = int)
            for k in range(dataset.shape[0]):
                class_prob = np.zeros([self.num_class],dtype=int)
                for i in range(self.num_class):
                    likeli = 0
                    for j in range(self.m):
                        likeli += dataset[k,j]*np.log(self.theta[i,j])+(1-dataset[k,j])*np.log(1-self.theta[i,j])
                    class_prob[i]=likeli+math.log((self.N[i]+1)/(self.n+2))
 
                decision[k] = np.argmax(class_prob)
        return decision