__author__ = 'qdpercy'

import os
import scipy.io
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

class Data(object):
    def __init__(self, xtr, ytr, xte, yte):
        self.xtr = xtr
        self.xte = xte
        self.ytr = ytr
        self.yte = yte


class MySVM(object):

    def __init__(self, n, lmda=0.01, gm=1, kernel='rbf', nsweep=1000, batchsize=1):
        self.batchsize = batchsize
        self.alpha = np.zeros(n)
        self.lmda = lmda
        self.c = 1.0 / (n * lmda)
        self.dim = n
        self.gm = gm
        self.nsweep = nsweep
        self.T = nsweep * n - 1
        self.kernel = kernel
        self._obj = []
        self.nnz = []
        self.err_tr = []
        self.ker_oper = []  #number of kernel operation
        self.err_te = []
        self.has_kte = False

    def train(self, xtr, ytr):
        pass

    def train_test(self, xtr, ytr, xte, yte):
        pass

    def test(self, x, y):
        pass

    def set_train_kernel(self, xtr):
        if self.kernel == 'rbf':
            std = np.std(xtr, axis=0)
            x = xtr / std[np.newaxis, :]
            xsquare = np.sum(x ** 2, 1)
            xxT = np.dot(x, x.T)
            dist = xsquare[:, np.newaxis] - 2 * xxT + xsquare[np.newaxis, :]
            K = np.exp(-self.gm * dist)
        else:
            print "the other kernel tbd"
        self.ktr = K

    def set_test_kernel(self, xtr, xte):
        if self.kernel == 'rbf':
            std = np.std(xtr, axis=0)
            xtr = xtr / std[np.newaxis, :]
            xte = xte / std[np.newaxis, :]
            s1 = np.sum(xte**2, 1)
            s3 = np.sum(xtr**2, 1)
            s2 = np.dot(xte, xtr.T)
            dist = s1[:, np.newaxis] - 2 * s2 + s3[np.newaxis, :]
            kte = np.exp(-self.gm * dist)
        else:
            print "the other kernel tbd"
        self.kte = kte

    def _prox_mapping(self, g, x0, r):
        """
        proximal coordinate gradient mapping
        argmin  x*g + 1/r*D(x0,x)
        """
        x = x0 - r * g
        x = np.minimum(np.maximum(0, x), self.c)
        return x
