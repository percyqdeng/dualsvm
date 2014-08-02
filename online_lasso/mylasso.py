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
    :param: algo: 'scg', stochastic coordinate gradient dual averaging;
                    'rda', regularized stochastic gradient dual averaging, Lin Xiao's NIPS paper
                    'cd', coordinate descent, shooting algorithm
    :param:    T: total number of iteration

    """
    def __init__(self, lmda=.1, b=1, c=5, T=1000, algo='scg'):
        self.lmda = lmda
        self.w = None
        self.T = T
        self.num_iters = None
        self.num_features = None
        self.num_zs = None
        self.obj = None
        self.sqnorm_w = None
        self.algo = algo
        if algo == 'scg':
            self.b = b
            self.c = c

    def fit(self, xtrain, ytrain):
        n, p = xtrain.shape
        if self.algo == 'scg':
            self.w, self.obj, self.num_zs, self.num_iters, self.num_features, self.sqnorm_w= \
                scg_lasso.train(x=xtrain, y=ytrain, b=np.int(self.b), c=np.int(self.c), lmda=np.uint64(self.lmda), T=np.uint64(self.T))
        elif self.algo == 'rda':
            self.w, self.obj, self.num_zs, self.num_iters, self.num_features, self.sqnorm_w= \
                rda_lasso.train(x=xtrain, y=ytrain, lmda=self.lmda, T=np.intp(self.T))
        elif self.algo == 'cd':
            self.w, self.obj, self.num_zs, self.num_iters, self.num_features, self.sqnorm_w= \
                cd_lasso.train(x=xtrain, y=ytrain, lmda=self.lmda, T=np.intp(self.T))
        else:
            print "algorithm type: \n" \
                  "scg', stochastic coordinate gradient dual averaging;\n" \
                  " 'rda', regularized stochastic gradient dual averaging, Lin Xiao's NIPS paper\n" \
                  " 'cd', coordinate descent, shooting algorithm"
            sys.exit(1)

    def predict(self, xtest):
        y = xtest.dot(self.w)
        return y
