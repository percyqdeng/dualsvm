__author__ = 'qdpercy'

import os
import scipy.io
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt


class MySVM(object):

    def __init__(self, n, lmda=0.01, gm=1, kernel='rbf', nsweep=1000, batchsize=1):
        self.batchsize = batchsize
        self.alpha = np.zeros(n)
        self.lmda = lmda
        self.num = n
        self.gm = gm
        self.nsweep = nsweep
        self.T = nsweep * n - 1
        self.kernel = kernel
        self.obj = []
        self.nnzs = []
        self.err_tr = []
        self.nker_opers = []  # number of kernel operation
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
            x = xtr
            xsquare = np.sum(x ** 2, 1)
            xxT = np.dot(x, x.T)
            dist = xsquare[:, np.newaxis] - 2 * xxT + xsquare[np.newaxis, :]
            K = np.exp(-self.gm * dist)
        else:
            print "the other kernel tbd"
        self.ktr = K

    def set_test_kernel(self, xtr, xte):
        if self.kernel == 'rbf':
            # std = np.std(xtr, axis=0)
            # xtr = xtr / std[np.newaxis, :]
            # xte = xte / std[np.newaxis, :]
            s1 = np.sum(xte**2, 1)
            s3 = np.sum(xtr**2, 1)
            s2 = np.dot(xte, xtr.T)
            dist = s1[:, np.newaxis] - 2 * s2 + s3[np.newaxis, :]
            kte = np.exp(-self.gm * dist)
        else:
            print "the other kernel tbd"
        self.kte = kte


