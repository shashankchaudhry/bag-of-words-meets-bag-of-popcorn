# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:59:45 2015

@author: schaud7
"""

from __future__ import division
from __future__ import print_function

import os
from pprint import pprint
from time import time
import logging

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from KaggleWord2VecUtility import KaggleWord2VecUtility
from ExtractScore import ExtractScore

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


if __name__ == '__main__':

    np.random.seed(0) # seed to shuffle the train set

    n_folds = 5
    verbose = True
    shuffle = False

    # Load data from the training set
    train_in = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
    unlab_train_in = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'unlabeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
    test_in = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
               quoting=3 )
    y = train_in["sentiment"]
    y = y.values
    
    print("Cleaning and parsing movie reviews...\n")
    traindata = []
    for i in xrange( 0, len(train_in["review"])):
        traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train_in["review"][i], True, False, False)))
    unlab_traindata = []
    for i in xrange( 0, len(unlab_train_in["review"])):
        score = ExtractScore.get_score_from_string(unlab_train_in["review"][i])
        if(score != None):
            if(score == 1.0):
                y = np.append(y,[1])
            else:
                y = np.append(y,[0])
            traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(unlab_train_in["review"][i], True, False, False)))
        else:
            unlab_traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(unlab_train_in["review"][i], True, False, False)))
    testdata = []
    for i in xrange(0,len(test_in["review"])):
        testdata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test_in["review"][i], True, False, False)))
    
    print("Converting to TFIDF features...\n")
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
    train = X_all[:lentrain]
    train_unlab = X_all[lentrain:lentrain + lenunlabtrain] 
    test = X_all[lentrain + lenunlabtrain:]

    if shuffle:
        idx = np.random.permutation(y.size)
        train = train[idx]
        y = y[idx]

    print("Creating n folds...\n")
    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            #GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)
            ]

    dataset_blend_train = np.zeros((train.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((test.shape[0], len(clfs)))
    
    print("Making classifier based data...\n")
    
    for j, clf in enumerate(clfs):
        print(j, clf)
        dataset_blend_test_j = np.zeros((test.shape[0], len(skf)))
        for i, (train_val, test_val) in enumerate(skf):
            print("Fold", i)
            X_train = train[train_val]
            y_train = y[train_val]
            X_test = train[test_val]
            y_test = y[test_val]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test_val, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(test)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
    
    print("Blending")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]
    
    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    
    output = pd.DataFrame( data={"id":test_in["id"], "sentiment":y_submission} )
    
    # Use pandas to write the comma-separated output file
    output.to_csv('/Users/schaud7/Documents/personal_git/data/my_eleventh_model_with_ensemble.csv', index=False, quoting=3)
    print ("Wrote results to my_eleventh_model_with_ensemble.csv")