# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 09:31:45 2021

@author: KX764QE
"""

import string
import nltk
import os
SEED_VALUE = 101
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
import random
random.seed(SEED_VALUE)
import numpy as np
np.random.seed(SEED_VALUE)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

TEST_SIZE = 0.3

BATCH_SIZE = 15000
NUM_BATCHES = 20

EVAL_BATCH_SIZE = 124777

ALL_CLASSES = [str(i) for i in range(1419)]
PUNCTUATION_REMOVAL = [punct for punct in list(string.punctuation) if punct != '.']

#resource: https://gist.github.com/sebleier/554280
STOPWORD_CUST_LIST = ['yeah','ah','uh','um','oh','all', 'me', 'i', 'my', 'myself', 
                      'we', 'our', 'you', "you're", "you've", "you'll", "you'd", 
                      'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                      'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
                      "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
                      'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                      "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 
                      'be', 'been', 'a', 'an', 'the', 'and', 'but', 'because', 'as',
                      'until', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                      'between', 'into', 'through', 'during', 'before', 'after', 
                      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 
                      'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                      'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 
                      'most', 'other', 'some', 'such', 'same', 'so', 'than', 's', 't']