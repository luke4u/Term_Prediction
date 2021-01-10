# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 09:34:26 2021

@author: KX764QE
"""

import config
import data_preprocessing
import os
import math
import random
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from nltk import FreqDist
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

def create_term_population(df):
    """
    This is to create a population of terms for each label
    
    Arguments:
        df {[dataframe]} -- [cleaned terms after punntuations and stop words removal]
    
    Outputs:
        vocabs {dictionary[list]} -- [key : label value: term population]
    """
    vocabs = defaultdict(list)
    for _, row in df.iterrows():
        label = row['label']
        term = row['term_no_sw']
        vocabs[label].append(term)
    for label, term_list in vocabs.items():
        vocabs[label] = ' '.join(term_list)
    return vocabs

def term_frequency(raw_txt):
    """
    This is to calculate the term frequency distribution for each label
    
    Arguments:
        raw_txt {[str]} -- [term pupulation per label]
    
    Outputs:
        fd {} -- [Count of each each token in the population]
        
    """
    text_tokens = word_tokenize(raw_txt)
    fd = FreqDist(text_tokens)
    return fd

def idf(dataset, word):
    """
    This is to calculate the inverse document frequency distribution for each label
    
    Arguments:
        dataset {[dataframe} -- [term pupulation per label]
        word {[string]} -- [word of interest of idf]
    
    Outputs:
        inv_df {[float]} -- [idf score of the word]
        
    """
    
    count = dataset['text'].apply(lambda x: word in x)
    inv_df = math.log(len(count)/sum(count))
    return inv_df


def tf_idf(dataset):
  """
  This is to calculate tf-idf score and retain words with score more than the mean of all words in a label
  
  Future work: 
      analyse the high TF-IDF word in terms of percentage of Noun, Verb, Adv, Adj
      
  Arguments:
        dataset {[dataframe} -- [term pupulation per label]
    
    Outputs:
        tfidf_high_freq_dict {[dictionary - list of tuples]} -- [high frequent word and its tf-idf score per label]
  
  """
  tfidf_high_freq_dict = {}

  for index, row in dataset.iterrows():
      if index % 100 == 0:
          print('   Progess: {}/{}'.format(index+1,dataset.shape[0]))
      term_scores = {}
      file_fd = term_frequency(row['text'])
      for word in file_fd:
          #skip non-alphabet words and only on Noun and Verb
          if word.isalpha() and data_preprocessing.get_wordnet_pos(word) in (wordnet.NOUN, wordnet.VERB):
              idf_val = idf(dataset,word)
              tf_val = term_frequency(row['text'])[word]
              tfidf_val = tf_val * idf_val
              term_scores[word] = round(tfidf_val, 2)
      
      term_scores_sorted = sorted(term_scores.items(), key=lambda x:-x[1])
      tfidf_high_freq_dict[str(row['label'])] = [(word, score) for word, score in term_scores_sorted if score > np.mean(list(term_scores.values()))]
    
  return tfidf_high_freq_dict

def enrich_term(df_in, tf_idf_score_in, enrich_count = 2):
  """
  This is to enrich each term with randomly picked high frequent words from each label vocab.
  This is likely to overfit the model.
  
  Future work: 
      reduce code runtime
      split data into train and test
      only enrich train set to test overfitting
      
  Arguments:
        df_in {[dataframe} -- [raw input data]
        tf_idf_score_in {[dictionary - list of tuples} -- [high frequent word and its tf-idf score per label]
        enrich_count {[int]} -- [count of words picked for each term]
    Outputs:
        df_in {[dataframe} -- [enriched data]
  
  """
  for index, row in df_in.iterrows():
    label = row['label']
    term = row['term']
    tf_idf_score_label = tf_idf_score_in[str(label)]
    word_label = [word for word, _ in tf_idf_score_label]
    picks = random.choices(word_label, k=enrich_count)
    df_in.loc[index, 'term_enriched'] = term + ' ' + ' '.join(picks)
  
  return df_in

if __name__ == '__main__':
    """
    This is to enrich the term by considering the frequent Noun and Verb in each term population vocab
    """
    pathtosave='../outputs/data/'
    if not os.path.exists(pathtosave):
        os.makedirs(pathtosave)
    filename_enriched = pathtosave +'trainSet_enriched.csv'     
    if not os.path.exists(filename_enriched):
        
        cols = ['term', 'label']
        df = pd.read_csv('../trainSet.csv', header=None, names = cols)
    
        
        df['term'] = [data_preprocessing.remove_punctuation(term, config.PUNCTUATION_REMOVAL)  
                      for term in df['term']]
        
        df['term_no_sw'] = [data_preprocessing.remove_stopwords(term, stopwords_list='customized', 
                                             cust_stopword=config.STOPWORD_CUST_LIST) 
                            for term in df['term']]
        df['term_no_sw'] = [data_preprocessing.lemmatize(term) for term in df['term_no_sw']]
        
        filename_vocab = pathtosave +'vocab.csv'
        if not os.path.exists(filename_vocab):
            vocabs = create_term_population(df)
            vocabs_df = pd.DataFrame(vocabs.items(), columns=['label', 'text'])
            vocabs_df.to_csv(filename_vocab)
        else:
            vocabs_df = pd.read_csv(filename_vocab)
            
        filename_tfidf = pathtosave +'tf_idf.json'
        if not os.path.exists(filename_tfidf):
            tf_idf_score = tf_idf(vocabs_df)
            with open(filename_tfidf, 'w') as fp:
                json.dump(tf_idf_score, fp)
        else:
            with open(filename_tfidf, 'r') as json_file:
                tf_idf_score = json.load(json_file)
                
        df_train, df_eval = train_test_split(df, test_size = config.TEST_SIZE, shuffle = True, 
                                             stratify = df['label'], random_state = config.SEED_VALUE)
        filename_train = pathtosave +'trainSet_train.csv'
        filename_eval = pathtosave +'trainSet_eval.csv'
        if not os.path.exists(filename_train):
            df_train.to_csv(filename_train)
            df_eval.to_csv(filename_eval)
        
        
        df_enriched = enrich_term(df_train, tf_idf_score, enrich_count= 2)
        df_enriched.to_csv(filename_enriched)
