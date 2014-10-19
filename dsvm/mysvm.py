__author__ = 'qdpercy'

import os
import scipy.io
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from kernel_func import KernelFunc
from rand_dataset import RandomDataset


class MySVM(object):
    def __init__(self, lmda=0.01, gm=1.0, deg=0, kernelstr='rbf', nsweep=1, b=5, c=1):
        self.b = b
        self.c = c
        self.lmda = lmda
        self.gm = gm
        self.degree = deg
        self.nsweep = nsweep
        self.alpha = None
        self.obj = []
        self.nnzs = []
        self.err_tr = []
        self.nker_opers = []  # number of kernel operation
        self.err_te = []
        self.ker_oper = []
        self.ktr = None
        self.kte = None
        self.xtr = None
        self.ytr = None
        self.kernel = []
        self.dataset = None
        self.kernelstr = kernelstr
        if self.kernelstr == 'rbf':
            self.kernel = KernelFunc(gamma=self.gm, ktype=1)
        elif self.kernelstr == 'poly':
            self.kernel = KernelFunc(degree=self.degree, ktype=2)
        else:
            raise NotImplementedError("undefined kernel type")

    def fit(self, xtr, ytr, xte, yte):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def construct_dataset(self, xtr, ytr, xte, yte):

        n = xtr.shape[0]
        if n > 2000:
            self.dataset = RandomDataset(xtrain=xtr, ytrain=ytr, kernel_function=self.kernel, xtest=xte, ytest=yte)
        else:
            ktr = self.kernel_matrix(xtr)
            if xte is not None:
                kte = self.kernel_matrix(xte, xtr)
            else:
                kte = None
            self.dataset = RandomDataset(xtrain=ktr, ytrain=ytr, kernel_function=self.kernel, xtest=kte, ytest=yte)

    def kernel_matrix(self, x1, x2=None):
        if self.kernelstr == 'rbf':
            x1square = np.sum(x1 ** 2, 1)
            if x2 is None:
                x2 = x1
                x2square = x1square
            else:
                x2square = np.sum(x2 ** 2, 1)
            x1x2T = np.dot(x1, x2.T)
            dist = x1square[:, np.newaxis] - 2 * x1x2T + x2square[np.newaxis, :]
            K = np.exp(-self.gm * dist)
        else:
            raise NotImplementedError("undefined kernel type")
        return K

