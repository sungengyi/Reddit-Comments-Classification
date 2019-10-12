
import time
import numpy as np


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
import NaiveBayes as nb

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

    nb_samples = 20000
    nb_rounds = 1
    x = np.zeros((nb_rounds))
    y = np.zeros((nb_rounds))
    start_time = time.time()
    for i in range(nb_rounds):
        bnbdata_X, bnbdata_Y = make_classification(n_samples=nb_samples, n_features=1000, n_informative=1000,n_classes=2, n_redundant=0)
        
        binarize(bnbdata_X)
        bnb = MultinomialNB()
        y_pred2 = bnb.fit(bnbdata_X,bnbdata_Y).predict(bnbdata_X)
        '''
        mnb = nb.NaiveBayes(num_class=2)
        mnb.fit(bnbdata_X,bnbdata_Y)
        y_pred3 = mnb.predict(bnbdata_X)
        
        #print(y_pred3)
        print("mnb: ",(bnbdata_Y != y_pred3).sum(),"bnb: ",(bnbdata_Y != y_pred2).sum())
        y[i] =(bnbdata_Y != y_pred2).sum()
        x[i] =(bnbdata_Y != y_pred3).sum()
        '''
    finish_time = time.time()
    print("-----Processed in {} sec".format(finish_time - start_time))
    return np.var(x),np.var(y),np.average(x),np.average(y)

a,b,c,d = test()



