# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand
from libc.math cimport fmax
from libc.math cimport fmin

def gen_rand_int(int n=500000):
    cdef int i
    cdef int ind
    cdef double res
    for i in xrange(1, n):
        ind = int(rand() % n)
        res = float(ind)/n
        res = fmin(res, n/2)

def gen_rand_int2(int n=500000):
    cdef int i
    cdef int ind
    cdef double res
    for i in xrange(1, n):
        ind = int(rand() % n)
        res = float(ind)/n
        res = cy_min(res, n/2)

def mat_access(int n = 1000):
    cdef double[:,::1] mat = np.zeros((n,n))
    cdef int i, j
    for i in xrange(n):
        for j in xrange(n):
            rand()
            rand()
            mat[i, j] = rand() % 10

def mat_access2(int n = 1000):
    cdef double[:,::1] mat = np.zeros((n,n))
    cdef int i, j, k
    for k in xrange(n**2):
        i = rand()%n
        j = rand()%n
        mat[i, j] = rand() % 10

def mat_access3(int n = 1000):
    cdef double[:,::1] mat = np.zeros((n,n))
    cdef int i, j
    for i in xrange(n):
        for j in xrange(n):
            rand()
            rand()
            mat[j, i] = rand() % 10

def mat_access4(int n = 1000):
    cdef np.ndarray[double, dim=2, mode='c'] mat = np.zeros((n, n))
    cdef unsigned int i, j
    for i in xrange(n):
        for j in xrange(n):
            rand()
            rand()
            mat[i, j] = rand() % 10

def mat_access4(int n = 1000):
    cdef double ** mat = new double*[n]
    for i in xrange(n):
        mat[i] = new double[n]

    for i in xrange(n):
        for j in xrange(n):
            mat[i,j] = rand() % 10


cdef inline double cy_min(double a, double b):
    return a if a<b else b
