# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:17:04 2019

@author: zhouh
"""
import csv
import string
import numpy as np
import pandas as pd

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


class fileloader(object):
    def __init__(self):
        self.data = None
        
    def read(self,name):
        self.data = pd.read_csv('../data/original_data/%d.csv'%(name))
    
    def get(self,do_index=False):
        return self.data.to_csv(index=do_index)
    
class filefilter(object):

    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.sns = SnowballStemmer("english", ignore_stopwords=True)
    def load(self,data):
        self.data = data
    def split(self):
        self.data_split = []
        for w in self.data:
            self.data_split.insert(w.split())
        for w in self.data_split:
            w = [x for x in w if not(x.isnumeric() or x is in stopwords)]
            w = [x for x in w if not x is ]
        