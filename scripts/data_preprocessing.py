# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 09:32:42 2021

@author: KX764QE
"""

import config
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import stem, pos_tag
from nltk.corpus import wordnet
# nlp=spacy.load('en_core_web_sm')
# from spacy.lang.en.stop_words import STOP_WORDS
# from nltk.corpus import stopwords as stopwords_nltk


def remove_punctuation(term, symbols):
    """
    This is to remove punctuation found above
    
    Arguments:
        term {[string]} -- [word of interest for punct removal]
        symbols {[list[string]]} -- [list of puncts]
    Outputs:
        term {[string]} -- [word of interest after punct removal]
    """
    for punct in symbols:
        term = term.replace(punct, '')
    return term

def remove_stopwords(term, stopwords_list='customized', **kwargs):
    """
    This is to remove stop words, using customized stopwords list
    
    Arguments:
        term {[string]} -- [term of interest for punct removal]
        stopwords_list {[string]]} -- [tag to specify which stop words list to use]
        **kwargs {[dictionary]} - [keyword arguments for customized stop words list]
        
    Outputs:
        raw_txt {[string]} -- [term after stop words removal]
    """
    text_tokens = word_tokenize(term)
    if stopwords_list=='spacy':
        tokens_without_sw = [word for word in text_tokens if not word in STOP_WORDS]
    
    elif stopwords_list=='nltk':
        tokens_without_sw = [word for word in text_tokens if not word in stopwords_nltk]
    
    elif stopwords_list=='customized':
        cust_stopwords=kwargs['cust_stopword']
        tokens_without_sw = [word for word in text_tokens if not word in cust_stopwords]
    
    raw_txt = " ".join(tokens_without_sw)

    return raw_txt

def get_wordnet_pos(word):
    """
    This is to Map POS tag for each word
    Arguments:
        word {[string]} -- [word of interest]
    
    Outputs:
        t {[wordnet tag]} -- [wordnet tag type ]
    """
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    t = tag_dict.get(tag, wordnet.NOUN)
    return t

def lemmatize(txt, lemmatizer='nltk'):
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
        lemma_tokens = [wordnet_lemmatizer.lemmatize(x,get_wordnet_pos(x)) for x in text_tokens]
        txt_ = " ".join(lemma_tokens)
    return txt_

if __name__ == '__main__':
    """
    This is to preprocess data for code verification purpose
    """
    cols = ['term', 'label']
    df = pd.read_csv('../trainSet.csv', header=None, names = cols)

    
    df['term'] = [remove_punctuation(term, config.PUNCTUATION_REMOVAL)  
                  for term in df['term']]
    
    
    df['term_no_sw'] = [remove_stopwords(term, stopwords_list='customized', 
                                         cust_stopword=config.STOPWORD_CUST_LIST) 
                        for term in df['term']]
    
    df['term_no_sw'] = [lemmatize(term) for term in df['term_no_sw']]
    
    print(df.head())
    
    
    
    
    
    
    
    