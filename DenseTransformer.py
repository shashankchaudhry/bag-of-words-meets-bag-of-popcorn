# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:40:49 2015

@author: schaud7
"""
from sklearn.base import TransformerMixin

class DenseTransformer(TransformerMixin):

    def get_params(self, deep=True):
        return {}
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self