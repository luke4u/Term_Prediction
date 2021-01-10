# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 09:36:34 2021

@author: KX764QE
"""

import re
import config
from nltk import stem, pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessingTransform(BaseEstimator, TransformerMixin):
    
    def __init__(self, removestopwords=True,stopwords_list='customized',
                 cust_stopword=config.STOPWORD_CUST_LIST, remove_punctuation=True, 
                 cust_punctuation=config.PUNCTUATION_REMOVAL, lemmatization=True):
        #Stopwords
        self.removestopwords=removestopwords
        self.stopwords_list=stopwords_list
        self.cust_stopword = cust_stopword
        #REMOVE PUNCTUATION
        self.remove_punctuation=remove_punctuation
        self.lemmatization = lemmatization
        
    def removing_stopwords(self, raw_txt, stopwords_list, cust_stopword):
        """
        This is to remove customized stopwords
        Arguments:
            raw_txt {string} -- [search term]
            stopwords_list {string} -- [stop words list source tag]
            cust_stopword {[string]} -- [stopwords list]
        
        Outputs:
            raw_txt {string} -- [stopwords removed search term]
        """
        text_tokens = word_tokenize(raw_txt)
        if stopwords_list=='spacy':
            tokens_without_sw = [word for word in text_tokens if not word in STOP_WORDS]

        elif stopwords_list=='nltk':
            tokens_without_sw = [word for word in text_tokens if not word in stopwords_nltk]

        elif stopwords_list=='customized':
            
            tokens_without_sw = [word for word in text_tokens if not word in config.STOPWORD_CUST_LIST]

        raw_txt = " ".join(tokens_without_sw)
        return raw_txt
    
    def removing_punctuation(self, raw_txt, symbols):
        """
        This is to remove punctuation
        Arguments:
            word {string} -- [search term]
            symbols {[string]} -- [punct list]
        
        Outputs:
            raw_txt {string} -- [punct removed search term]
        """
        for punct in symbols:
            raw_txt = raw_txt.replace(punct, '')
        return raw_txt
    
    def get_wordnet_pos(self, word):
        """
        This is to Map POS tag for each word
        Arguments:
            word {string} -- [word of interest]
        
        Outputs:
            t {wordnet tag} -- [wordnet tag type ]
        """
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        t = tag_dict.get(tag, wordnet.NOUN)
        return t

    def lemmatize(self, txt, lemmatizer='nltk'):
        """
        This is to lemmatize a term based on token pos
        Arguments:
            word {string} -- [search term]
        
        Outputs:
            txt_ {string} -- [lemmatized search term]
        """
        if lemmatizer=='spacy':
            doc=nlp(txt)
            txt_=" ".join([str(w.lemma_) if w.lemma_!= '-PRON-' else w.text for w in doc])
        elif lemmatizer=='nltk':
            wordnet_lemmatizer = stem.WordNetLemmatizer()
            text_tokens = word_tokenize(txt)            
            lemma_tokens = [wordnet_lemmatizer.lemmatize(x,self.get_wordnet_pos(x)) for x in text_tokens]
            txt_ = " ".join(lemma_tokens)
        return txt_

    def transform(self, X, y=None):
        X_term=X.values
        if self.removestopwords:
            X_term=[self.removing_stopwords(term,stopwords_list=self.stopwords_list, 
                                            cust_stopword=self.cust_stopword) for term in X_term]
        
        if self.remove_punctuation:
            X_term=[self.removing_punctuation(term, config.PUNCTUATION_REMOVAL) for term in X_term]
        
        if self.lemmatization:
            X_term=[self.lemmatize(term) for term in X_term]
        
        
        X_term=[re.sub(' +',' ',sent) for sent in X_term]
        
        return X_term
    
    def fit(self, df, y=None):
        """
        Returns self unless something different happens in train and test
        """
        return self
