# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 09:47:06 2021

@author: KX764QE
"""


import sys
import pickle
import time
import numpy as np
import pandas as pd

def predict_documents(folder_name = 'candidateTestSet.txt', path='outputs/ml_classification/'):
    """
    predict term

    Arguments:
        foler_name {str} -- [path to holdout dataset]
        path {str} -- [path to save model results] (default: {'outputs/ml_classification/'})
        
    Outputs:
        df_test {[dataframe]} -- [prediction results]
    """

    try:
        print('loading model')
        loaded_model = pickle.load(open(path+'final_models_'+'.sav', 'rb'))

    except Exception as E:
        print(E)
        print('No model available, train before you can predict.')
        return
    cols = ['term']
    df_test = pd.read_csv(folder_name, header=None, names = cols)
    #Get Predictions of the model
    print('Doing predictions')
    df_test['classification'] = loaded_model.predict(df_test['term'])
    print('predictions done')
    return df_test
    

if __name__ == '__main__':
    """
    This is to predict testset.
    """
    start = time.time()
    try:
        folder_name_test_path = sys.argv[1]
        df_test = predict_documents(folder_name = folder_name_test_path, path='outputs/ml_classification/')
        df_test.to_csv('prediction_results.csv')
    except IndexError:
        print('Please add test dataset file path.')
    
    diff = time.time() - start
    print('Prediction time: ', np.round(diff, 2))
                      
    