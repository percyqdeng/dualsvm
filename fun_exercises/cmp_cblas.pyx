# distutils: language = c++

# toy example to demonstrate cblas function in cython

# cimport cython
import numpy as np
cimport numpy as np
# import time
#
cdef extern from "cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor=101
        CblasColMajor=102
    enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
        AtlasConj=114
    void daxpy "cblas_daxpy"(int N, double alpha, double *X, int incX,
                             double *Y, int incY) nogil
    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY
                             ) nogil
    double dasum "cblas_dasum"(int N, double *X, int incX) nogil
    void dger "cblas_dger"(CBLAS_ORDER Order, int M, int N, double alpha,
                double *X, int incX, double *Y, int incY, double *A, int lda) nogil
    void dgemv "cblas_dgemv"(CBLAS_ORDER Order,
                      CBLAS_TRANSPOSE TransA, int M, int N,
                      double alpha, double *A, int lda,
                      double *X, int incX, double beta,
                      double *Y, int incY) nogil


cpdef ddot_blas(double [:]a, double [:] b):
    cdef Py_ssize_t n = a.size
    cdef double res
    res = ddot(n, &a[0], 1, &b[0], 1)
    return res


cpdef ddot_cy(double [:]a, double[:] b):
    cdef Py_ssize_t i, n = a.size
    cdef double res = 0
    for i in xrange(n):
        res += a[i] * b[i]
    return res

# @cython.boundscheck(False)
# @cython.cdivision(True)
# @cython.wraparound(False)
cpdef mat_vec_blas(double[:, ::1]aa, double[::1]b):
    cdef Py_ssize_t m, n
    cdef double[::1] c = np.zeros(n)
    n = aa.shape[0]
    m = aa.shape[1]
    # start_t = time.time()
    dgemv(CblasRowMajor, CblasNoTrans, n, m, 1, &aa[0,0], n, &b[0], 1, 1, &c[0], 1)
    # print "mat_vec_blass time %f" % (time.time()-start_t)

# @cython.boundscheck(False)
# @cython.cdivision(True)
# @cython.wraparound(False)
# cpdef mat_vec_cy(double[:,::1]aa, double[::1]b):
#     """
#     c = aa * b
#     :param aa:
#     :param b:
#     :param c:
#     :return:
#     """
#     cdef int n = aa.shape[0]
#     cdef int m = aa.shape[1]
#     cdef double [::1] c = np.zeros(n)
#     assert (m == n)
#     cdef int i, j
#     # start_t = time.time()
#     for i in xrange(n):
#         c[i] = 0
#         for j in xrange(m):
#             c[i] += aa[i,j] * b[j]
#     # print "mat_vec_cy time %f" % (time.time()-start_t)

