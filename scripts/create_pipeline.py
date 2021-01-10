# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 09:42:18 2021

@author: KX764QE
"""


import config
import numpy as np
import PreprocessingTransformer as pp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

def create_pipelines(**kwargs):
    
    removestopwords=kwargs.get('removestopwords',True)
    stopwords_list=kwargs.get('stopwords_list','customized')
    cust_stopword=kwargs.get('cust_stopword', config.STOPWORD_CUST_LIST)
    removepunctuation=kwargs.get('removepunctuation', True)
    cust_punctuation=kwargs.get('cust_punctuation', config.PUNCTUATION_REMOVAL)
    lemmatization=kwargs.get('lemmatization',False) 
    print('kwargs: \n', kwargs)
    
    #Initialization
    preprocessingtransf=pp.PreprocessingTransform(removestopwords=removestopwords,
                                                stopwords_list=stopwords_list, 
                                                cust_stopword=cust_stopword, 
                                                remove_punctuation=True, 
                                                cust_punctuation=cust_punctuation,
                                                lemmatization=lemmatization)
    vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=True, lowercase=True) # dtype = np.float16
    
    clf_nb=MultinomialNB()
    # clf_gb=GradientBoostingClassifier(random_state=config.SEED_VALUE)
    
    dict_gridsearchpipelines={}
    
    dict_gridsearchpipelines['multinomialnb']={
        'pipeline': Pipeline([
                ('feats', Pipeline([
                    ('PreprocessingTransform', preprocessingtransf),
                    ('vectorizer', vectorizer),
                ])
                ),
              ('clf', clf_nb)]),
        'params':{
    'feats__vectorizer__ngram_range':[(1,1),(1,2), (1,3)],
    # 'feats__vectorizer__norm': ['l1', 'l2'],
    # 'feats__vectorizer__binary':[True, False]
    }}
    """
    
    dict_gridsearchpipelines['grandientboosting']={
        'pipeline': Pipeline([
                ('feats', Pipeline([
                    ('PreprocessingTransform', preprocessingtransf),
                    ('vectorizer', vectorizer),
                ])
                ),
              ('clf_gb', clf_gb)]),
        'params':{
             'feats__vectorizer__ngram_range':[(1,2)],
            # 'feats__vectorizer__norm': ['l1', 'l2'],
            'clf_gb__n_estimators':[200],
            #'clf_gb__criterion':["friedman_mse", "mse", "mae"],
    }}
    """
    return dict_gridsearchpipelines