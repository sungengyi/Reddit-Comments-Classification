# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 22:23:43 2019

@author: sunge
"""
import nltk
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from tqdm import tqdm
from numpy import transpose as T
from scipy.stats import stats
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
# 2.0   Count word Frequency in each subreddit
#       Generating all frequency files
#------------------------------------------------------------------------------
# 2.1 Load file
#------------------------------------------------------------------------------
file_name_df = pd.read_csv('../data/subreddits.csv')
lemmatizer = WordNetLemmatizer() 


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]

for name in file_name_df['0']:
    file_df = pd.read_csv('../data/parsed_data/{}.csv'.format(name))
    print("---------------------------File Loaded-----------------------------")
   
    # 2.1 Clean comments
    #------------------------------------------------------------------------------
    stop_words = set(stopwords.words('english')) 
    stemmer = SnowballStemmer("english")

    file_df['delete_symbol_token'] = file_df['comments'].str.replace('[{}]'.format(string.punctuation), '')
    file_df['delete_stopword_token']= file_df['delete_symbol_token'].str.lower().apply(lambda x: [item for item in str(x).split() if item not in stop_words])
    file_df['text_lemmatized'] = file_df.delete_stopword_token.apply(lambda x : [lemmatizer.lemmatize(w) for w in x])
    print("--------------------------Data Processed---------------------------")

   # 2.2 Get unique words in comments
    #------------------------------------------------------------------------------
    i = 0
    list_of_word = []
    for word in file_df['text_lemmatized']:
        # Run through the list checking for existeing word
        for element in list_of_word :
            if word == element:
                i+=1
        if i == 0:
            list_of_word.append(word)
        else:
            i = 0
    print("----------------------Same word in same row------------------------")

            
    new_list_of_words = []      
    for word in list_of_word:
        new_list_of_words = new_list_of_words + word
           
    print("-----------------------------Adding--------------------------------")

    # 2.3 Concade all unique words in one list
    #------------------------------------------------------------------------------
    i = 0
    new_unique_list = []
    for word in new_list_of_words:
        # Run through the list checking for existeing word
        for element in new_unique_list :
            if word == element:
                i+=1
        if i == 0:
            new_unique_list.append(word)
        else:
            i = 0
    print("----------------------------Same word------------------------------")

    # 2.4 Concade all words in one list
    #------------------------------------------------------------------------------
    new_word_list = []
    for word in file_df['text_lemmatized']:
        new_word_list = new_word_list + word
    
    word_df = pd.DataFrame(data = new_word_list)
    wordcount_df = word_df.stack().value_counts().to_frame('occurrence').loc[new_unique_list] 
    # 2.5 Write in csv files
    #------------------------------------------------------------------------------
    print("----------------------------Writing---------------------------------")

    wordcount_df.to_csv(r'../data/frequencey_data/{}-word-occurrence.csv'.format(name),',')





















