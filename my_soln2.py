# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:05:59 2015

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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


###############################################################################
# Load data from the training set
train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
               quoting=3 )
y = train["sentiment"]

#clean the input data
print("Cleaning and parsing movie reviews...\n")
traindata = []
for i in xrange( 0, len(train["review"])):
    traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))
testdata = []
for i in xrange(0,len(test["review"])):
    testdata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2), max_features=20000, max_df=0.5)),
    ('tfidf', TfidfTransformer(norm='l2', use_idf=True)),
    ('boost', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)))
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'boost__n_estimators': (500, 300)
}
# result: 500 was better

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, scoring='roc_auc')

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(traindata, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    # getting results with the best parameters
    #vectorizer = CountVectorizer(max_df=0.5, max_features=None, ngram_range=(1,2))
    
#    newpipeline = Pipeline([
#        ('vect', CountVectorizer(max_df=0.5, max_features=None, ngram_range=(1, 2))),
#        ('tfidf', TfidfTransformer(norm='l2', use_idf=True)),
#        ('clf', SGDClassifier(alpha=1e-05, n_iter=80, penalty='l2', loss='log')),
#    ])
    
#    newpipeline.fit(traindata, y)
#    
#    result = newpipeline.predict_proba(testdata)[:,1]
#    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
#
#    # Use pandas to write the comma-separated output file
#    output.to_csv('/Users/schaud7/Documents/personal_git/data/my_first_model.csv', index=False, quoting=3)
#    print ("Wrote results to my_first_model.csv")