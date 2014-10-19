# distutils: language = c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fmax, fmin, exp
"""
kernel type
1: rbf
2: poly
"""
cdef class KernelFunc(object):
    def __init__(self, double gamma=1, int degree=1, int ktype=1):
        self.gamma = gamma
        self.degree = degree
        self.ktype = ktype


    cdef double product(self, double[::1]x1, double[::1] x2):
        cdef Py_ssize_t i, j, p = x1.shape[0]
        cdef double res = 0
        if self.ktype == RBF:
            for i in xrange(p):
                res += (x1[i] - x2[i]) *(x1[i] - x2[i])
            res *= - self.gamma
            res = exp(res)
        elif self.ktype == POLY:
            pass
        return res

    # cpdef kernel_matrix(self, x1, x2):
    #     cdef Py_ssize_t n, m, i, j
    #     n = x1.shape[0]
    #     if x2 is None:
    #         m = n
    #     else:
    #         m = x2.shape[0]
    #
    #     # K = np.zeros((n, m), dtype=float, ord='C')
    #     if self.ktype == 1:
    #         x1square = np.sum(x1 ** 2, 1)
    #         x2square = np.sum(x2**2, 1)
    #         x1x2T = np.dot(x1, x2.T)
    #         dist = x1square[:, np.newaxis] - 2 * x1x2T + x2square[np.newaxis, :]
    #         K = np.exp(-self.gm * dist)
    #     else:
    #         print "the other kernel tbd"
    #         K = None
    #     return K