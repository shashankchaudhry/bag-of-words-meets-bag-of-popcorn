# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:34:12 2015

@author: schaud7
"""

from __future__ import print_function

from pprint import pprint
from time import time
import logging

import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from ExtractScore import ExtractScore

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


###############################################################################
# Load data from the training set
train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
unlab_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'unlabeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
               quoting=3 )
y = train["sentiment"]

count = 0
bad_count = 0
for i in xrange(len(train["review"])):
    score_sen = ExtractScore.get_score_from_string(train["review"][i])
    if(score_sen != None):
        new_sen = 0
        if(score_sen == 1.0):
            new_sen = 1
        else:
            new_sen = 0
        if(new_sen != y[i]):
            print(train["review"][i])
            ExtractScore.get_score_from_string(train["review"][i])
            print(score_sen)
            print(y[i])
            print()
            bad_count = bad_count + 1
        else:
            count = count + 1
print(count)
print(bad_count)
print()

#clean the input data
print("Cleaning and parsing movie reviews...\n")
traindata = []
for i in xrange( 0, len(train["review"])):
    traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True, False, False)))
testdata = []
for i in xrange(0,len(test["review"])):
    testdata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True, False, False)))
    
    
# insights: Some people use smileys, some say grade: f etc. grade could be ** out of *****
#newpipeline = Pipeline([
#        ('vect', CountVectorizer(max_df=0.5, max_features=None, ngram_range=(1, 2))),
#        ('tfidf', TfidfTransformer(norm='l2', use_idf=True)),
#        ('clf', SGDClassifier(alpha=1e-05, n_iter=80, penalty='l2', loss='log')),
#    ])
#    
#newpipeline.fit(traindata, y)
#    
#result = newpipeline.predict_proba(testdata)[:,1]
#output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
#
## Use pandas to write the comma-separated output file
#output.to_csv('/Users/schaud7/Documents/personal_git/data/my_fifth_model.csv', index=False, quoting=3)
#print ("Wrote results to my_fourth_model.csv")