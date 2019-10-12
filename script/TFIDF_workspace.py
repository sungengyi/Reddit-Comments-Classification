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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import NaiveBayes as nb
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB

#------------------------------------------------------------------------------
# 1.1 Load file
#------------------------------------------------------------------------------
#file_name_df = pd.read_csv('../data/subreddits.csv')
lemmatizer = WordNetLemmatizer() 
stemmer = SnowballStemmer("english")

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]

#for name in file_name_df['0']:
    
'''
file_dfidf = pd.read_csv('../data/original_data/reddit_train.csv')
print(file_dfidf.shape)
'''
# 2.1 Clean comments
#------------------------------------------------------------------------------
stop_words = set(stopwords.words('english')) 
stemmer = SnowballStemmer("english")
'''
file_dfidf['delete_symbol_token'] = file_dfidf['comments'].str.replace('[{}]'.format(string.punctuation), '')
file_dfidf['delete_stopword_token']= file_dfidf['delete_symbol_token'].str.lower().apply(lambda x: [item for item in str(x).split() if item not in stop_words])
file_dfidf['text_lemmatized'] = file_dfidf.delete_stopword_token.apply(lambda x : [lemmatizer.lemmatize(w) for w in x])
file_dfidf['text_stemmized'] = file_dfidf['text_lemmatized'].apply(lambda x : [stemmer.stem(w) for w in x])
'''
'''
corpus = file_dfidf['comments']
vectorizer1 = TfidfVectorizer(stop_words=stop_words)
X1 = vectorizer1.fit_transform(corpus)
wnl = WordNetLemmatizer()
snss = SnowballStemmer("english", ignore_stopwords=True)
my_stop_words = [snss.stem(wnl.lemmatize(t)) for t in stopwords.words('english')]

'''
    
    


'''
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.sns = SnowballStemmer("english", ignore_stopwords=True)
    def __call__(self, articles):
        articles.split();
        


vectorizer2 = TfidfVectorizer(tokenizer = LemmaTokenizer(),stop_words=my_stop_words)
X2 = vectorizer2.fit_transform(corpus)
'''
def binarize(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j]>0:
                X[i,j]=1
            else:
                X[i,j]=0

def test():
    diff = 0
    nb_samples = 3000
    for i in range(100):
        bnbdata_X, bnbdata_Y = make_classification(n_samples=nb_samples, n_features=8, n_informative=4, n_redundant=0)
        binarize(bnbdata_X)
        bnb = MultinomialNB()
        y_pred2 = bnb.fit(bnbdata_X,bnbdata_Y).predict(bnbdata_X)
        mnb = nb.NaiveBayes()
        mnb.fit(bnbdata_X,bnbdata_Y)
        y_pred3 = mnb.predict(bnbdata_X)
        print("mnb: ",(bnbdata_Y != y_pred3).sum(),"bnb: ",(bnbdata_Y != y_pred2).sum())
        if ((bnbdata_Y != y_pred3).sum()!=(bnbdata_Y != y_pred2).sum()):
            diff+=1
    return diff

print("test error rate out of 100 round: %d" %(test()))



