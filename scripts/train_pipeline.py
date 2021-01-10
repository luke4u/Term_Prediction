# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 09:44:02 2021

@author: KX764QE
"""

import sys
import os
import time
import pandas as pd
import operator
import pickle
import config
import numpy as np
from create_pipeline import create_pipelines
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import warnings
warnings.filterwarnings("ignore")

def ml_pipeline(col, targetvar, pathtosave='outputs/ml_classification/', **kwargs):
    """
    Training a model (gridsearch based) and training it for later usage

    Arguments:
        col {[str]} -- [Column where the text of the model]
        targetvar {[[binary numeric]} -- [target variable]

        pathtosave {str} -- [path to save outputs] (default: {'outputs/ml_classification/'})
    """
    if not os.path.exists(pathtosave):
        os.makedirs(pathtosave)
        
    X_train, X_test, y_train, y_test = train_test_split(col, targetvar, 
                                                        test_size=0.25, 
                                                        random_state = config.SEED_VALUE)
    # print(X_train.shape, y_train.shape)
    #Pipeline & Gridsearch
    #scoring_m=['accuracy','precision_macro', 'recall_macro', 'f1_macro']    
    gridsearchcv=create_pipelines(**kwargs)
    
    for model in gridsearchcv.keys():
        print('------------------------------------', model, '-------------------------------------------')
        gscv=GridSearchCV(gridsearchcv[model]['pipeline'], param_grid=gridsearchcv[model]['params'],  
                          n_jobs=-1, verbose=3, cv=2, refit='precision_macro')   
        gridsearchcv[model]['fitted_model']=gscv.fit(X_train, y_train)
        print('Finish training %s!' %model)
    
    model_decision={}
     #Check prediction on unseen evidences
    for model in gridsearchcv.keys():
        testpredictions=gridsearchcv[model]['fitted_model'].best_estimator_.predict(X_test)
        report=classification_report(y_test, testpredictions, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        

        filename_train_metrics = pathtosave + model +'_train_metrics.csv'
        df_report.to_csv(filename_train_metrics)
        model_decision[model]=df_report.loc['macro avg', 'precision']
    
    final_model=max(model_decision.items(), key=operator.itemgetter(1))[0]
    
    #Save the model to disk
    filename = pathtosave +'final_models_'+'.sav'
    pickle.dump(gridsearchcv[final_model]['fitted_model'].best_estimator_, open(filename, 'wb'))
    # print('Best estimator: ', gridsearchcv[final_model]['fitted_model'].best_estimator_)
    
    return

def train_models(folder_name = 'outputs/data/trainSet_enriched.csv', **kwargs):
    """
    Arguments:
        foler_name {str} -- [path to enriched train dataset]
        removestopwords {bool} -- [default as True to remove stopwords] 
        remove_punctuation {bool} -- [default as True to remove puncts] 
        
    """
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    cols = ['term_enriched', 'label'] #term_enriched
    raw_data = pd.read_csv(folder_name, usecols=cols)
    result = ml_pipeline(raw_data['term_enriched'], raw_data['label'], 
                         pathtosave='outputs/ml_classification/', **kwargs)
    
    return



if __name__ == '__main__':
    """
    This is to train the models using grid search.
    Multinominal naive bayes and Gradient Boost classifier are used for grid search
    
    Future work:
            convert tf-idf vector data type from float64 to float8 to reduce memeory usage and runtime
    """
    
    start = time.time()
    
    try:
        folder_name_train_path = sys.argv[1]
        train_models(folder_name = folder_name_train_path, stopwords_list='customized', 
                 cust_stopword=config.STOPWORD_CUST_LIST, cust_punctuation=config.PUNCTUATION_REMOVAL,
                 removestopwords=True, remove_punctuation=True, lemmatization=False)
    except IndexError:
        print('Please add train dataset file path.')

    diff = time.time() - start
    print('Train time: ', np.round(diff, 2))
    

