import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import scg_lasso


class LassoLR(object):
    """
    online lasso
    """
    def __init__(self, lmda=.1, b=1, c=5, T=1000):
        self.b = b
        self.c = c
        self.lmda = lmda
        self.w = None
        self.T = T
        self.num_iters = None
        self.num_features = None
        self.num_zs = None
        self.obj = None

    def fit(self, xtrain, ytrain):
        n, p = xtrain.shape
        self.w, self.obj, self.num_zs, self.num_iters, self.num_features = \
            scg_lasso.train(x=xtrain, y=ytrain, b=np.int(self.b), c=np.int(self.c), lmda=self.lmda, T=np.uint64(self.T))

    def predict(self, xtest):
        y = xtest.dot(self.w)
        return y
