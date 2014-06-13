# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand
from libc.math cimport fmax
from libc.math cimport fmin

cpdef gen_rand_int(int n=500000):
    cdef int i
    cdef int ind
    cdef double res
    for i in xrange(1, n):
        ind = int(rand() % n)
        res = float(ind)/n
        res = fmin(res, n/2)

cpdef gen_rand_int2(int n=500000):
    cdef int i
    cdef int ind
    cdef double res
    for i in xrange(1, n):
        ind = int(rand() % n)
        res = float(ind)/n
        res = cy_min(res, n/2)


cdef inline double cy_min(double a, double b):
    return a if a<b else b