# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: qdeng

cimport cython
from libc.limits cimport INT_MAX
cimport numpy as np
import numpy as np
from kernel_func cimport KernelFunc

np.import_array()


cdef class RandomDataset(object):
    def __init__(self, double[:,::1] xtrain, double[::1]ytrain, KernelFunc kernel_function, int precomputed=1,
                double[:,::1] xtest=None, double[::1]ytest=None, ):
        self.is_precomputed = precomputed
        self.kernel_function = kernel_function
        self.ytrain = ytrain
        self.n_train = ytrain.shape[0]
        self.ytest = ytest
        if xtest is None:
            self.has_test_data = 0
            self.n_test = 0
        else:
            self.has_test_data = 1
            self.n_test = xtest.shape[0]

        if self.is_precomputed:
            self.ktrain = xtrain
            self.ktest = xtest
        else:
            self.xtrain = xtrain
            self.xtest = xtest
        # self.n_train = xtrain.shape[0]
        # self.n_dims = ytrain.shape[0]
        # print "n_train %f, n_dims %f" % (xtrain.shape[1], ytrain.shape[0])
        # assert( xtrain.shape[0] == self.n_dims)

        # if self.n_train < 2000:
        #     self.is_precomputed = 1
        #     self.ktrain = np.zeros((self.n_train, self.n_train))
        #     K = kernel_function.kernel_matrix(xtrain, xtrain,)
        #     self.ktrain = K
        #     if xtest is not None:
        #         self.ktest = np.zeros((self.n_test, self.n_train))
        #         K = kernel_function.kernel_matrix(xtest, xtrain)
        #         self.ktest = K
        #         self.n_test = xtest.shape[0]
        # else:
        #     self.is_precomputed = 0


    cdef double kernel_entry(self, int i, int j):
        if self.is_precomputed:
            return self.ktrain[i, j]
        else:
            return self.kernel_function.product(self.xtrain[i], self.xtrain[j])


    cdef void ktrain_dot_yalpha(self, double[::1]alpha, double [::1]z):
        """
        the classifier output on test data
        z <- k.dot(y*alpha)
        """
        cdef Py_ssize_t n, i, j
        n = self.n_train
        if self.is_precomputed:
            for i in xrange(n):
                z[i] = 0
                for j in xrange(n):
                    z[i] += self.ktrain[i, j] * self.ytrain[j] * alpha[j]
        else:
            for i in xrange(n):
                z[i] = 0
                for j in xrange(n):
                    z[i] += self.kernel_function.product(self.xtrain[i], self.xtrain[j]) * self.ytrain[j] * alpha[j]


    cdef void ktest_dot_yalpha(self, double[::1]alpha, double [::1]z):
        """
        the classifier output on test data
        z <- k.dot(y*alpha)
        """
        cdef Py_ssize_t i, j, n, m
        n = self.n_test
        m = self.n_train
        if self.is_precomputed:
            for i in xrange(n):
                z[i] = 0
                for j in xrange(m):
                    z[i] += self.ktest[i, j] * self.ytrain[j] * alpha[j]
        else:
            for i in xrange(self.n_test):
                z[i] = 0
                for j in xrange(self.n_train):
                    z[i] += self.kernel_function.product(self.xtest[i], self.xtrain[j]) * self.ytrain[j] * alpha[j]
