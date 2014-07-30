import numpy as np
cimport numpy as np
from libc.stdlib cimport rand

"""
lasso problem:
0.5||y-xw||^2 + lmda * ||w||_1
the code implements the algorithm in page 443, "Machine Learning" by Kevin Murphy
"""
def train(double [:,::1] x, int[::1]y, int b=1, int c=2, double lmda=0.1, Py_ssize_t T=100):
    cdef Py_ssize_t i, j, t, p, n
    n = x.shape[0]
    p = x.shape[1]
    cdef double[::1] w = np.zeros(p)
    cdef double[::1] z = np.zeros(n)
    cdef double[::1] a = np.zeros(p)
    cdef double c
    for i in xrange(n):
        for j in xrange(p):
            a[j] += x[i,j] * x[i, j]

    for t in xrange(T):
        j = rand() % p
        c = 0
        for i in xrange(p):
            pass



def train_test(double [:,::1] x, int[::1]y, double[:,::1]xtest, int[:]ytest,
                int b=1, int c=2, double lmda=0.1, Py_ssize_t T=100):
    """
    coordinate descent for lasso
    :return:
    """