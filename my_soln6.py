# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:09:18 2015

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
import scipy as sp

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

#clean the input data
print("Cleaning and parsing movie reviews...\n")
traindata = []
for i in xrange( 0, len(train["review"])):
    traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True, False, False)))
unlab_traindata = []
for i in xrange( 0, len(unlab_train["review"])):
    score = ExtractScore.get_score_from_string(unlab_train["review"][i])
    if(score != None):
        if(score == 1.0):
            y = y.append(pd.Series([1]))
        else:
            y = y.append(pd.Series([0]))
        traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(unlab_train["review"][i], True, False, False)))
    else:
        unlab_traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(unlab_train["review"][i], True, False, False)))
testdata = []
for i in xrange(0,len(test["review"])):
    testdata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True, False, False)))
    
# get all the data
X_all = traindata + unlab_traindata + testdata
lentrain = len(traindata)
lenunlabtrain = len(unlab_traindata)
lentestdata = len(testdata)
    
newpipeline = Pipeline([
        ('vect', CountVectorizer(max_df=0.5, max_features=None, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(norm='l2', use_idf=True)),
    ])
    
newpipeline.fit(X_all)
X_all = newpipeline.transform(X_all)
X = X_all[:lentrain]
X_unlab = X_all[lentrain:lentrain + lenunlabtrain] 
X_test = X_all[lentrain + lenunlabtrain:]
print(len(X_test))

model = SGDClassifier(alpha=1e-05, n_iter=150, penalty='l2', loss='log')
print("Retrain on all training data, predicting test labels...\n")
model.fit(X,y)

X_first_set = sp.sparse.vstack([X_unlab,X_test])
result = model.predict_proba(X_first_set)[:,1]
# select the pieces where result probability is greater than 0.95 or less than 0.05
X_subset = X_first_set[(result < 0.05) | (result > 0.95)]
result = result[(result < 0.05) | (result > 0.95)]

# put result back into y
X = sp.sparse.vstack([X,X_subset])
for res in result:
    if(res < 0.05):
        y = y.append(pd.Series([0]))
    else:
        y = y.append(pd.Series([1]))
model.fit(X,y)
result = model.predict_proba(X_test)[:,1]
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv('/Users/schaud7/Documents/personal_git/data/my_ninth_model_with_extract_score_bigrams_overall_tfidf_super_learning.csv', index=False, quoting=3)
print ("Wrote results to my_fourth_model.csv")