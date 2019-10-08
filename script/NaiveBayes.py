import numpy as np
import math

class NaiveBayes:
    def __init__(self, ls = True):
        self.ls = ls

        
        
    def fit(self, X_features, Y_quality):
        self.X_features = X_features
        self.Y_outcomes = Y_quality
        self.n,self.m = self.X_features.shape
        N0 = 0
        N1 = 0
        for i in range(self.Y_outcomes.shape[0]):
            if self.Y_outcomes[i] == 0:
                N0 += 1
            else:
                N1 += 1
        self.N = np.array([N0,N1])
        
        self.w = np.zeros([2,self.m],dtype=int)
        self.theta = np.ones_like(self.w,dtype = float)
        self.ww = np.ones_like(self.w,dtype = float)
        # make array w that stores all features p(xi|y=c)
        for j in range (self.m):
            for i in range(self.n):
                if self.X_features[i,j] == 1:
                    if self.Y_outcomes[i]== 0:
                        self.w[0,j]=self.w[0,j]+1
                    else:
                        self.w[1,j]=self.w[1,j]+1
                        
        for j in range(self.m):
            for i in range(2):
                self.theta[i,j]=self.w[i,j]/self.N[i]
        if self.ls == True :
            self.laplace_smooth()
        for j in range(self.m):
                self.ww[0,j] = math.log((1-self.theta[1,j])/(1-self.theta[0,j]))
                self.ww[1,j] = math.log((self.theta[1,j])/(self.theta[0,j]))
                
 
               
    def laplace_smooth(self):
        for j in range(self.m):
            for i in range(2):
                self.theta[i,j]=(self.w[i,j]+1)/(self.N[i]+2)

        
    def predict(self, dataset):
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
        