# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

stemmer = SnowballStemmer("english")

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer() 


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]


df = pd.DataFrame(['this was cheesy', 'she likes these books', 'wow this is great'], columns=['text'])
df['text_lemmatized'] = df.text.apply(lemmatize_text)

file_df = pd.read_csv('../data/parsed_data/hockey.csv')
stop_words = set(stopwords.words('english')) 
file_df['delete_symbol_token'] = file_df['comments'].str.replace('[{}]'.format(string.punctuation), '')
file_df['delete_stopword_token']= file_df['delete_symbol_token'].str.lower().apply(lambda x: [item for item in str(x).split() if item not in stop_words])
file_df['text_lemmatized'] = file_df.delete_stopword_token.apply(lemmatize_text)
#lemmanized_token = delete_stopword_token.apply(lambda x : [lemmatize_text(w) for w in str(x).split])
