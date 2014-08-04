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
                    'rda2', simple adaptation of rda in limited information setting
                    'cd', coordinate descent, shooting algorithm
    :param:    T: total number of iteration
    :param eta: learnning rate
    """
    def __init__(self, lmda=.1, b=1, c=5, T=1000, algo='scg', sig_D=1000.0):
        self.lmda = lmda
        self.w = None
        self.T = T
        self.num_iters = None
        self.num_features = None
        self.num_zs = None
        self.train_obj = None
        self.test_obj = None
        self.sqnorm_w = None
        self.algo = algo
        self.sig_D = sig_D
        if algo == 'scg' or algo == 'rda2':
            self.b = b
            self.c = c

    def fit(self, xtrain, ytrain, xtest=None, ytest=None):
        has_test = not (xtest is None)
        n, p = xtrain.shape
        if not has_test:
            if self.algo == 'scg':
                self.w, self.train_obj, self.num_zs, self.num_iters, self.num_features, self.sqnorm_w= \
                    scg_lasso.train(x=xtrain, y=ytrain, b=np.int(self.b), c=np.int(self.c), sig_D=self.sig_D, lmda=self.lmda, T=np.intp(self.T))
            elif self.algo == 'rda':
                self.w, self.train_obj, self.num_zs, self.num_iters, self.num_features, self.sqnorm_w= \
                    rda_lasso.train(x=xtrain, y=ytrain, sig_D=self.sig_D, lmda=self.lmda, T=np.intp(self.T))
            elif self.algo == 'rda2':
                self.w, self.train_obj, self.num_zs, self.num_iters, self.num_features, self.sqnorm_w= \
                    rda_lasso.train2(x=xtrain, y=ytrain, sig_D=self.sig_D, lmda=self.lmda, T=np.intp(self.T))
            elif self.algo == 'cd':
                self.w, self.train_obj, self.num_zs, self.num_iters, self.num_features, self.sqnorm_w= \
                    cd_lasso.train(x=xtrain, y=ytrain, lmda=self.lmda, T=np.intp(self.T))
            else:
                print "algorithm type: \n" \
                      "scg', stochastic coordinate gradient dual averaging;\n" \
                      " 'rda', regularized stochastic gradient dual averaging, Lin Xiao's NIPS paper\n" \
                      " 'cd', coordinate descent, shooting algorithm"
                sys.exit(1)
        else:
            if self.algo == 'scg':
                self.w, self.train_obj, self.test_obj, self.num_zs, self.num_iters, self.num_features, self.sqnorm_w= \
                    scg_lasso.train(x=xtrain, y=ytrain, xtest=xtest, ytest=ytest, b=np.int(self.b), c=np.int(self.c), lmda=self.lmda, sig_D=self.sig_D, T=np.intp(self.T))
            elif self.algo == 'rda':
                self.w, self.train_obj, self.test_obj, self.num_zs, self.num_iters, self.num_features, self.sqnorm_w= \
                    rda_lasso.train(x=xtrain, y=ytrain, xtest=xtest, ytest=ytest,  lmda=self.lmda, sig_D=self.sig_D, T=np.intp(self.T))
            elif self.algo == 'rda2':
                self.w, self.train_obj, self.test_obj, self.num_zs, self.num_iters, self.num_features, self.sqnorm_w= \
                    rda_lasso.train2(x=xtrain, y=ytrain, xtest=xtest, ytest=ytest, b=np.int(self.b), c=np.int(self.c), lmda=self.lmda, sig_D=self.sig_D, T=np.intp(self.T))
            elif self.algo == 'cd':
                self.w, self.train_obj, self.test_obj, self.num_zs, self.num_iters, self.num_features, self.sqnorm_w= \
                    cd_lasso.train(x=xtrain, y=ytrain, xtest=xtest, ytest=ytest, lmda=self.lmda, T=np.intp(self.T))
            else:
                print "algorithm type: \n" \
                      "scg', stochastic coordinate gradient dual averaging;\n" \
                      " 'rda', regularized stochastic gradient dual averaging, Lin Xiao's NIPS paper\n" \
                      " 'cd', coordinate descent, shooting algorithm"
                sys.exit(1)

    def eval_lasso_obj(self, x, y, lmda):
        n, p = x.shape
        res = 0.5/n * np.sum((y-x.dot(self.w))**2) + lmda * np.linalg.norm(self.w, ord=1)
        return res

    def predict(self, xtest):
        y = xtest.dot(self.w)
        return y

    def esti_learn_rate(self, n, p):
        # learning rate is std of gradient, of l2 norm of gradient,
        self.sig_D = self.eta * np.sqrt(p**3) / np.sqrt(p)
