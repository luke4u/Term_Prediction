# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:53:08 2021

@author: KX764QE
"""

#----https://scikit-learn.org/0.15/modules/scaling_strategies.html
import sys
import os
import time
import pandas as pd
import operator
import pickle
import config
import data_preprocessing as dp
import matplotlib.pyplot as plt
import numpy as np
from stream_create_pipeline import create_pipelines
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score
import warnings
warnings.filterwarnings("ignore")
#https://stackoverflow.com/questions/57109473/gridsearchcv-randomizedsearchcv-with-partial-fit-in-sklearn

def stream_files(path):
    """
    This is to stream train set line by line.
    Arguments:
        path: {str} -- [path to train set data]
    
    """
    with open(path, 'r', encoding="utf8") as train:
        for text in train:
            yield text
            
def get_batch(doc_stream, bat_size):
    """
    This is to stream train set batch by batch. Note below is specific for trainSet_enriched.csv on the index
    Arguments:
        doc_stream: {iterator} -- [train set data iterator]
        bat_size: {int} -- [batch size]
        
    Returns:
        X, y: {[str, int]} -- [list of raw terms and labels]
    """
    X, y = [], []
    for _ in range(bat_size):
        try:
            text = next(doc_stream)
        except Exception as e:
            print('Warning: Reached the end of the file!')
            break
        line = text.strip().split(',')
        X.append(line[-1])
        y.append(line[2])
        
    return X, y      
      
def get_eval_batch(func, doc_stream, eval_bat_size):
    """
    This is stream eval batch
    Arguments:
        func {function object} -- [file stream function]
        doc_stream: {iterator} -- [train set data iterator]
        eval_bat_size: {int} -- [evaluation batch size]
    Returns:
        X_eval_batch, y_eval_batch: {[str, int]} -- [list of raw terms and labels]
    """
    X_eval_batch, y_eval_batch = func(doc_stream, eval_bat_size)
    return X_eval_batch, y_eval_batch

def iter_batches(doc_iter, batch_size):
    """Generator of batches."""
    X, y = get_batch(doc_iter, batch_size)
    while len(X):
        yield X, y
        X, y = get_batch(doc_iter, batch_size)
        
def progress(clf_name, stats):
    """
        Report progress information, return a string.
    """
    duration = time.time() - stats['t0']
    s = "%20s classifier : \t" % clf_name
    # s += "%(n_train)6d train lines (%(n_train_pos)6d positive) " % stats
    # s += "%(n_test)6d test lines (%(n_test_pos)6d positive) " % test_stats
    s += "macro_precision: %(macro_precision).3f " % stats
    s += "in %.2fs (%4d lines/s )" % (duration, stats['n_train'] / duration)
    return s

def plot_accuracy(x, y, x_legend):
    """
        Plot accuracy as a function of x.
    """
    x = np.array(x)
    y = np.array(y)
    plt.title('Classification accuracy as a function of %s' % x_legend)
    plt.xlabel('%s' % x_legend)
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(x, y)
    
def ml_pipeline(vectorizer_path, train_file_path,  pathtosave='outputs/ml_classification/', **kwargs):
    """
    Training a model (gridsearch based) and training it for later usage
    Arguments:
        vectorizer_path {str} -- [path to fitted vectorizer] (default: {'outputs/ml_classification/'})
        pathtosave {str} -- [path to save outputs] (default: {'outputs/ml_classification/'})
    """
    if not os.path.exists(pathtosave):
        os.makedirs(pathtosave)
    
    partial_fit_classifiers = {
    'SGD': SGDClassifier(n_jobs=-1),
    'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier(n_jobs=-1),
    }
    # print(os.getcwd())
    with open('./outputs/ml_classification/vectorizer.pk', 'rb') as vect:
        vectorizer = pickle.load(vect)
            
    file_stream = stream_files(train_file_path)
    X_eval, y_eval = get_eval_batch(get_batch, file_stream, config.EVAL_BATCH_SIZE)
    X_eval = dp.preprcoessing_pipeline(X_eval)
    X_eval_vects = vectorizer.transform(X_eval)
    batch_iterators = iter_batches(file_stream, config.BATCH_SIZE)
    test_stats = {'n_test': 0}
    test_stats['n_test'] += len(y_eval)
    # test_stats['n_test_pos'] += sum(y_eval)
    #Pipeline & Gridsearch
    #scoring_m=['accuracy','precision_macro', 'recall_macro', 'f1_macro']    
    gridsearchcv=create_pipelines(**kwargs)
    clf_stats = {}
    for clf_name in partial_fit_classifiers.keys():
        stats = {'n_train': 0, 'macro_precsion': 0.0, 
                 'precison_history': [(0, 0)], 
                 't0': time.time(),
                 'runtime_history': [(0, 0)], 
                 'total_fit_time': 0.0
                 
                 }
        clf_stats[clf_name] = stats
    total_vect_time = 0.0
    for i, (X_train_batch, y_train_batch) in enumerate(batch_iterators):
        
        tick = time.time()
        X_train_batch = dp.preprcoessing_pipeline(X_train_batch)
        X_train_batch_vects = vectorizer.transform(X_train_batch)
        total_vect_time += time.time() - tick
        
        for clf_name, clf in partial_fit_classifiers.items():
            # print('----------Batch Training on model %s---------------' %clf_name)     
            tick = time.time()
            clf.partial_fit(X_train_batch_vects, y_train_batch, classes=config.ALL_CLASSES)
            clf_stats[clf_name]['total_fit_time'] += time.time() - tick
            
            clf_stats[clf_name]['n_train'] += len(y_train_batch)
            
            tick = time.time()
            y_eval_pred = clf.predict(X_eval_vects)
            macro_precision = precision_score(y_eval, y_eval_pred, average = 'macro')
            clf_stats[clf_name]['macro_precision']=macro_precision
            clf_stats[clf_name]['prediction_time'] = time.time() - tick
            
            precision_history = (clf_stats[clf_name]['macro_precision'],
                       clf_stats[clf_name]['n_train'])
            clf_stats[clf_name]['precison_history'].append(precision_history)
            run_history = (clf_stats[clf_name]['macro_precision'], 
                           total_vect_time + clf_stats[clf_name]['total_fit_time'])
            clf_stats[clf_name]['runtime_history'].append(run_history)
            
            if i % 3 == 0:
                print('Batch %s'%(i+1), progress(clf_name, clf_stats[clf_name]))
            if i == 19:
                #Save the model to disk
                filename = pathtosave +'final_model_%s'%clf_name +'.pk'
                pickle.dump(clf, open(filename, 'wb'))
    print('-------------------Train completed----------------------------------')
    clf_names = list(sorted(clf_stats.keys()))
    plt.figure()
    for _, stats in sorted(clf_stats.items()):
        # Plot accuracy evolution with #examples
        precison, n_examples = zip(*stats['precison_history'])
        plot_accuracy(n_examples, precison, "training examples (#)")
        ax = plt.gca()
        ax.set_ylim((0.5, 1))
    plt.legend(clf_names, loc='best')
    plt.show()   
    return

def train_models(vectorizer_path, folder_name='outputs/data/trainSet_enriched.csv',  **kwargs):
    """
    Arguments:
        foler_name {str} -- [path to enriched train dataset]
        removestopwords {bool} -- [default as True to remove stopwords] 
        remove_punctuation {bool} -- [default as True to remove puncts] 
        
    """
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # cols = ['term_enriched', 'label'] #term_enriched
    # raw_data = pd.read_csv(folder_name, usecols=cols)
    result = ml_pipeline(vectorizer_path, folder_name, 
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
        vectorizer_path = r'../outputs/ml_classification/vectorizer.pk'
        train_models(vectorizer_path, folder_name = folder_name_train_path, stopwords_list='customized', 
                 cust_stopword=config.STOPWORD_CUST_LIST, cust_punctuation=config.PUNCTUATION_REMOVAL,
                 removestopwords=False, remove_punctuation=False, lemmatization=False)
    except IndexError:
        print('Please add train dataset file path.')

    diff = time.time() - start
    print('Train time: ', np.round(diff, 2))
    

