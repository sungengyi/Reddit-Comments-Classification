import time
import nltk
import itertools
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from numpy import transpose as T
from scipy.stats import stats
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

sns.set()
# 1.0 Processing Data
#------------------------------------------------------------------------------
# 1.1 Load file
#------------------------------------------------------------------------------
training_data_df = pd.read_csv('../data/original_data/reddit_train.csv')
# 1.2 Extract all subreddits
#------------------------------------------------------------------------------
list_of_subreddit = []
i = 0
for category in training_data_df['subreddits']:
    # Run through the list checking for existeing category
    for element in list_of_subreddit :
        if category == element:
            i+=1
    if i == 0:
        list_of_subreddit.append(category)
    else:
        i = 0
print(list_of_subreddit)
# Write to a file
subreddit_df = pd.DataFrame(data=list_of_subreddit)
subreddit_df.to_csv('../data/subreddits.csv', sep=',',index=False)
   
# 1.3 Count for occurances
#------------------------------------------------------------------------------
#Now count for occurances
#count_df = pd.DataFrame(0, index=np.arange(1),columns=list_of_subreddit)
#for category in training_data_df['subreddits']:
#    for column in count_df
count_df = training_data_df.stack().value_counts().to_frame('occurrence').loc[list_of_subreddit]  
    
# Divide the csv file according to subreddit
#------------------------------------------------------------------------------
'''
ONLY NEED TO RUN ONCE
'''
with open('../data/original_data/reddit_train.csv') as fin:    
    csvin = csv.DictReader(fin)
    # Category -> open file lookup
    outputs = {}
    for row in csvin:
        cat = row['subreddits']
        # Open a new file and write the header
        if cat not in outputs:
            fout = open('{}.csv'.format(cat), 'w')
            dw = csv.DictWriter(fout, fieldnames=csvin.fieldnames)
            dw.writeheader()
            outputs[cat] = fout, dw
        # Always write the row
        outputs[cat][1].writerow(row)
    # Close all the files
    for fout, _ in outputs.values():
        fout.close()  
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    