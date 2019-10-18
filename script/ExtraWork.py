# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:53:28 2019

@author: zhouh
"""

import csv
import numpy as np
import pandas as pd

#extra_data = pd.read_csv(r'../data/RC_2017-12-01.csv', error_bad_lines = False)

subreddit_df =  pd.read_csv(r'../data/subreddits.csv')

tdf = pd.DataFrame()

for element in subreddit_df['0']:
    df = pd.read_csv(r'../data/extra/{}_ext.csv'.format(element))
    tdf = tdf.append(df)
    