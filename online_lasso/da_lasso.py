import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import scg_lasso
import rda_lasso
import cd_lasso

class LassoLI(object):
    """
    online lasso
    :param: algo: 1, stochastic coordinate gradient dual averaging;
                    2, stochastic gradient dual averaging, Lin Xiao's NIPS paper
                    3, coordinate descent, shooting algorithm
    """
    def __init__(self, lmda=.1, b=1, c=5, T=1000, algo=1):
        self.lmda = lmda
        self.w = None
        self.T = T
        self.num_iters = None
        self.num_features = None
        self.num_zs = None
        self.obj = None
        self.algo = algo
        if algo == 1:
            self.b = b
            self.c = c

    def fit(self, xtrain, ytrain):
        n, p = xtrain.shape
        if self.algo == 1:
            self.w, self.obj, self.num_zs, self.num_iters, self.num_features = \
                scg_lasso.train(x=xtrain, y=ytrain, b=np.int(self.b), c=np.int(self.c), lmda=self.lmda, T=np.uint64(self.T))
        elif self.algo == 2:
            self.w, self.obj, self.num_zs, self.num_iters, self.num_features = \
                rda_lasso.train(x=xtrain, y=ytrain, lmda=(self.lmda), T=np.intp(self.T))
        elif self.algo == 3:

    def predict(self, xtest):
        y = xtest.dot(self.w)
        return y
