#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv('D:\\PGP-BABI\\Capstone\\Airbnb\\Tasmania-detailed reviews.csv', error_bad_lines=False)
#Drop rows with blank comments
data_valid = data.dropna()
#Group all comments for a single listing
data_grouped = data_valid.groupby('listing_id', sort=False)['comments'].apply(' '.join).reset_index()


# In[2]:


cluster_words=pd.read_csv('D:\\PGP-BABI\\Capstone\\Airbnb\\Clustering\\review\\review_cluster_CBOW_w3_spaCy_mincount_1000_10 - Upd.csv', error_bad_lines=False)


# In[3]:


#Create a list of distinct cluster values to loop through
cluster_values=list(set(cluster_words['cluster']))


# In[4]:


#Import libraries
import datetime
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
#import nltk
#nltk.download('punkt') #This is one time
import spacy
#import en_core_web_sm
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

#conda install rpy2 #Run in cmd prompt
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
utils = importr('utils')
#utils.install_packages('syuzhet') #This is one time
ro.r('library(syuzhet)')

#parallel processing
import numpy as np
import seaborn as sns
from multiprocessing import Pool


# In[5]:


#Create function to map sentences to features and assign sentiment score per feature
def map_sentence(row):
    result = pd.DataFrame([])
    text=row['comments']
    text=repr(text) # Remove non-unicode characters
    text=((text.encode('ascii','ignore')).decode("utf-8")).lower()
    
    # Remove new line characters
    text = text.replace("\\r\\n", " ")
    #text = text.replace("\\n\\n", " ")
    text = text.replace("\\n", " ")
    
    #Sentence tokenization
    for token in sent_tokenize(text):
        sent_cluster=[]
        temp=[]

        #Word tokenization
        tokenized_words=word_tokenize(token)

        #Lemmatization and POS tagging - Extract only Nouns and Adjectives
        allowed_postags=['NOUN', 'ADJ']
        doc = nlp(" ".join(tokenized_words)) 
        #for doc in nlp.pipe(" ".join(tokenized_words), batch_size=10000, n_threads=3):
            #pass
        temp.append([t.lemma_ for t in doc if t.pos_ in allowed_postags])
        temp=temp[0]

        #Sentence-to-feature one hot encoding
        for i in cluster_values:
            z=0
            for word in temp:
                if word in list(cluster_words.loc[cluster_words['cluster']==i]['word']):
                    z=z+1
            if z>0:
                sent_cluster.append(i)

        #Sentiment score using Vader
        score = analyser.polarity_scores(token)

        #Create data frame
        #df = pd.DataFrame({'listing_id':row['listing_id'],'sentence':token,'features':[sent_cluster],'pos_score':score['pos'],'neg_score':score['neg'],'neu_score':score['neu'],'compound_score':score['compound']})
        ro.globalenv["token"] = token
        df = pd.DataFrame({'listing_id':row['listing_id'],'sentence':token,'features':[sent_cluster],'score_nrc_pos':ro.r('get_nrc_sentiment(token)')[9],'score_nrc_neg':ro.r('get_nrc_sentiment(token)')[8],'score_overall':ro.r('get_sentiment(token)'),'pos_score_vader':score['pos'],'neg_score_vader':score['neg'],'neu_score_vader':score['neu'],'compound_score_vader':score['compound']})
        result = result.append(df)
        
    return result

#TODO: Avg sentiment score per feature + avg overall sentiment score, for each listing


# In[10]:


#Create a feature-sentiment-dataframe for all listings
test=data_grouped.iloc[0:50]
fea_sent_df = pd.DataFrame([])
print(datetime.datetime.now())
for index,row in test.iterrows():
#for index,row in data_grouped.iterrows():
    fea_sent_df = fea_sent_df.append(pd.DataFrame(map_sentence(row)))
print(datetime.datetime.now())
fea_sent_df.to_csv('D:\\PGP-BABI\\Capstone\\Airbnb\\Sentiment Scores\\50_listing_feature_sentiment-Upd.csv', sep=',')

