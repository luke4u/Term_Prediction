# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 09:46:33 2021

@author: KX764QE
"""


import sys
import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

def evaluate_models(folder_name = 'outputs/data/trainSet_eval.csv', pathtosave='outputs/ml_classification/'):
    """
    predict term

    Arguments:
        foler_name {str} -- [path to unenriched evaluation dataset]
        path {str} -- [path to save model results] (default: {'outputs/ml_classification/'})
        
    Outputs:
        df_test {[dataframe]} -- [prediction results]
    """
    
    try:
        print('loading model')
        loaded_model = pickle.load(open(pathtosave+'final_models_'+'.sav', 'rb'))

    except Exception as E:
        print(E)
        print('No model available, train before you can evaulate.')
        return
    if not os.path.exists('outputs/ml_classification/'):
        os.makedirs('outputs/ml_classification/')
    cols = ['term', 'label'] 
    df_eval = pd.read_csv(folder_name, usecols=cols)
    
    #Get Predictions of the model
    print('doing evaluation')
    df_eval['classification'] = loaded_model.predict(df_eval['term'])
    print('evaluation done')
    report=classification_report(df_eval['label'], df_eval['classification'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    filename = pathtosave +'eval_metrics.csv'
    df_report.to_csv(filename)
        
    return df_eval

if __name__ == '__main__':
    """
    This is to evaluate the model using unriched dataset
    """
    start = time.time()
    try:
        folder_name_eval_path = sys.argv[1]
        evaluate_models(folder_name = folder_name_eval_path, pathtosave='outputs/ml_classification/')
    except IndexError:
        print('Please add evaluation dataset file path.')
        
    diff = time.time() - start
    print('Evaluation time: ', np.round(diff, 2))