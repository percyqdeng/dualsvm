# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
import time
from libcpp.vector cimport vector
# from libcpp import bool
from libc.stdlib cimport rand
# cdef extern from "math.h"
from libc.math cimport fmax
from libc.math cimport fmin

ctypedef np.float64_t dtype_t
ctypedef np.int_t dtypei_t
from rand_no_replace import *


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def online_lasso_train(double [:,::1] x, double[::1]y, int b=1, int c=2, double lmda=0.1,  int T=1000):

    cdef unsigned int i, j, n, p, k, t
    n, p = x.shape
    cdef int[::1] I = np.zeros(c, dtype=np.int32)
    cdef int[::1] J = np.zeros(b, dtype=np.int32)
    cdef int[::1] u = np.zeros(p, dtype=np.int32)
    cdef double[::1] ind = np.zeros(1, dtype=np.int32)
    cdef double[::1] w_tilde = np.zeros(p)
    cdef double[::1] w = np.zeros(p)
    cdef double[::1] w_bar = np.zeros(p)
    cdef double cur_grad = np.zeros(p)
    cdef double[::1] accum_grad = np.zeros(p)  # accumulated stochastic coordinate gradient
    cdef double sig, D, gm
    sig = np.sqrt(p**3/c)
    D = p * 1
    gm = fmax(2*c, )
    r0 = RandNoRep(T)
    r1 = RandNoRep(n)
    r2 = RandNoRep(n)
    for t in xrange(T+1):
        ind = rand() % n
        r1.k_choice(I, c)
        r2.k_choice(J, b)
        for j in xrange(c):
            cur_grad = 0
            for i in xrange(b):
                cur_grad += x[ind, I[i]] * w[I[i]]
            cur_grad *= x[ind, J[j]] * p / c
            cur_grad -= y[ind] * x[ind, J[j]]
            accum_grad[J[j]] += cur_grad
            w_tilde[J[j]] += (t+1 - u[J[j]]) * w[J[j]]
            if accum_grad[J[j]] <= lmda:
                w[J[j]] = 0
            else:
                w[J[j]] = - (accum_grad[J[j]] - lmda*sign_func(accum_grad[J[j]]) ) / gm
            u[J[j]] = t + 1

    for j in xrange(p):
        w_bar[j] = w_tilde[j] + (T+1 - u[j])*w[j]

    return np.asarray(w_bar)

def online_train_validate
cdef inline sign_func(double x):
    if x >= 0:
        return 1
    else:
        return -1

