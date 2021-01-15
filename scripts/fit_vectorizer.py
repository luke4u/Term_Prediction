# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:35:06 2021

@author: KX764QE
"""
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd

def fit_vecterizer(df, col, dump_path):
    """
    This is to fit and dump trained vectorizer
    Arguments:
        df {dataframe} -- [train set dataframe]
        col {string} -- [train set column name]
    """
    vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=True, lowercase=True)
    vectorizer.fit(df[col])
    
    with open(dump_path, 'wb') as fin:
        pickle.dump(vectorizer, fin)
    

def load_trainset(filepath):
    """
    This is to load train set
    Arguments:
        filepath {string} -- [path to transet file]
    Return:
        df {dataframe} -- [train set dataframe]
    """
    cols = ['term_enriched', 'label'] #term_enriched
    df = pd.read_csv(filepath, usecols=cols)
    return df

if __name__ == '__main__':
    """
    This is to train vectorizer offline, to avoid memory overhead.
    """
    dump_path = '../outputs/ml_classification/vectorizer.pk'
    if not os.path.exists(dump_path):
        file_path = '../outputs/data/trainSet_enriched.csv'
        df = load_trainset(file_path)
        col = 'term_enriched'
        fit_vecterizer(df, col, dump_path)
    else:
        with open(dump_path, 'rb') as vect:
            vectorizer = pickle.load(vect)
        test_samples = ['the sky is blue and bright', 'the sub is bright']
        vects = vectorizer.transform(test_samples)
        print(vects.shape)
    

