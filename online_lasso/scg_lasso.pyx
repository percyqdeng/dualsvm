# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
import time
from libcpp.vector cimport vector
from libc.stdlib cimport rand
from libcpp cimport bool
from cpython cimport bool
# cdef extern from "math.h"
from libc.math cimport fmax
from libc.math cimport fabs
from libc.math cimport fmin
cdef extern from "time.h" nogil:
    ctypedef Py_ssize_t clock_t
    cdef clock_t clock()
    # int clock()
    cdef enum:
        CLOCKS_PER_SEC
ctypedef np.float64_t dtype_t
ctypedef np.int_t dtypei_t
# cimport rand_funcs

# cdef extern from "cblas.h":
#     enum CBLAS_ORDER:
#         CblasRowMajor=101
#         CblasColMajor=102
#     enum CBLAS_TRANSPOSE:
#         CblasNoTrans=111
#         CblasTrans=112
#         CblasConjTrans=113
#         AtlasConj=114
#
#     void daxpy "cblas_daxpy"(int N, double alpha, double *X, int incX,
#                              double *Y, int incY) nogil
#     double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY
#                              ) nogil
#     double dasum "cblas_dasum"(int N, double *X, int incX) nogil
#     void dger "cblas_dger"(CBLAS_ORDER Order, int M, int N, double alpha,
#                 double *X, int incX, double *Y, int incY, double *A, int lda) nogil
#     void dgemv "cblas_dgemv"(CBLAS_ORDER Order,
#                       CBLAS_TRANSPOSE TransA, int M, int N,
#                       double alpha, double *A, int lda,
#                       double *X, int incX, double beta,
#                       double *Y, int incY) nogil
#     double dnrm2 "cblas_dnrm2"(int N, double *X, int incX) nogil
#     void dcopy "cblas_dcopy"(int N, double *X, int incX, double *Y, int incY) nogil
#     void dscal "cblas_dscal"(int N, double alpha, double *X, int incX) nogil

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def train(double [:,::1] x, double[::1]y, double[:,::1]xtest=None, double[::1]ytest=None,
          int b=4, int c=1, double lmda=0.1, double sig_D=-1, int verbosity=True, Py_ssize_t T=1000):
    """
    online training lassolr, using stochastic coordinate gradient method
    :param x:
    :param y:
    :param b: batch size
    :param c: batch size
    :param lmda:
    :param T:
    :return: w_bar, train_res, num_zs, num_iter
    """
    cdef Py_ssize_t i, j, t
    cdef Py_ssize_t n, p
    n = x.shape[0]
    p = x.shape[1]
    assert(c<=p and b<=p)
    cdef Py_ssize_t [::1] I = np.zeros(c, dtype=np.intp)
    cdef Py_ssize_t [::1] J = np.zeros(b, dtype=np.intp)
    cdef Py_ssize_t [::1] u = np.zeros(p, dtype=np.intp)
    cdef double [::1] l = np.zeros(p)
    cdef int[::1] flag = np.ones(p, dtype=np.intc)  # flag to check zero pattern, flag[i]=1 if w[i]=0; 0 otherwise
    cdef int has_test = not (xtest is None)
    cdef double[::1] w_tilde = np.zeros(p)
    cdef double[::1] w = np.zeros(p)
    cdef double[::1] w_bar = np.zeros(p)
    cdef double cur_grad
    cdef double[::1] g_tilde = np.zeros(p)  # accumulated stochastic coordinate gradient
    cdef Py_ssize_t ind
    cdef double sig, D, gm
    cdef vector [double] num_iters
    cdef vector [double] num_features
    cdef vector [double] train_res
    cdef vector [double] test_res
    cdef vector [double] num_zs # number of zeros
    cdef vector [double] sqnorm_w
    cdef vector [double] timecost
    cdef Py_ssize_t num_steps=0, interval
    cdef Py_ssize_t count
    cdef Py_ssize_t start_t, end_t,
    cdef double cpu_t
    if T < 20:
        interval = 1
    else:
        interval = T / 20

    if not verbosity:
        interval = T
    cdef double tmp, tmp2, xw
    if sig_D < 0:
        sig = np.sqrt(p**3/c)
        D = np.sqrt(p * 1)
        sig_D = sig / D
    gm = fmax(2*c, np.sqrt(2.0*b*(T+1.0)/p)* sig_D)
    # print "---------------------scg lasso-------------------\n" \
    #       "lamda %f, gamma %f, b:%d, c:%d" % (lmda, gm, b, c)
    r1 = RandNoRep(p)
    r2 = RandNoRep(p)
    start_t = clock()
    for t in xrange(T+1):
        ind = rand() % n
        r1.k_choice(I, c)
        r2.k_choice(J, b)
        xw = 0
        for i in xrange(c):
            xw += x[ind, I[i]] * w[I[i]]
        for j in xrange(b):
            cur_grad = xw * x[ind, J[j]] * p / c - y[ind] * x[ind, J[j]]
            g_tilde[J[j]] += cur_grad
            w_tilde[J[j]] += (t+1 - u[J[j]]) * w[J[j]]
            l[J[j]] += lmda
            tmp2 = fabs(g_tilde[J[j]])
            flag[J[j]] = (tmp2<=l[J[j]])
            w[J[j]] = - sign_func(g_tilde[J[j]]) * fmax(tmp2-l[J[j]], 0) / gm
            u[J[j]] = t + 1
        if num_steps == t:
            end_t = clock()
            cpu_t += <double>(end_t - start_t) / CLOCKS_PER_SEC
            # print <double>(end_t - start_t) / CLOCKS_PER_SEC
            timecost.push_back(cpu_t)
            # compute train error, # of zeros
            num_iters.push_back(t)
            num_features.push_back((t+1) * (b+c))
            train_res.push_back(eval_lasso_obj(w, x, y, lmda))
            # interval *= 2
            num_steps += interval
            if has_test:
                test_res.push_back(eval_lasso_obj(w, xtest, ytest, lmda))
            count = 0
            tmp = 0
            for i in xrange(p):
                count += flag[i]
                tmp += w[i] * w[i]
            sqnorm_w.push_back(tmp)
            num_zs.push_back(count)
            start_t = clock()
    for j in xrange(p):
        w_bar[j] = (w_tilde[j] + (T+2 - u[j])*w[j]) / (T+1)
    # for j in xrange(p):
    #     w_bar[j] = w[j]
    if not has_test:
        return np.asarray(w_bar), train_res, num_zs, num_iters, num_features, sqnorm_w, timecost
    else:
        return np.asarray(w_bar), train_res, test_res, num_zs, num_iters, num_features, sqnorm_w, timecost


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef inline sign_func(double x):
    if x > 0:
        return 1
    elif x<0:
        return -1
    else:
        return 0

cdef inline soft_threshold(double a, double b, double c):
    """
    min 0.5*a*x^2 + b*x + c*|x|
    :return: -sign(b)/a *max(|b|-c,0)
    """
    return -sign_func(b) * fmax(fabs(b)-c, 0) / a



@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef eval_lasso_obj(double[::1]w, double[:,::1] x, double[::1]y, double lmda):
    """
    evaluate objective 0.5/n* ||Y-Xw||^2 + lmda * |w|
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t p = x.shape[1]
    cdef double [::1] z = np.zeros(n)
    mat_vec(x, w, z)
    cdef Py_ssize_t i
    cdef double res = 0
    for i in xrange(n):
        res += (y[i]-z[i])**2
    res *= 0.5 / n
    for i in xrange(p):
        res += lmda * fabs(w[i])
    return res


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef mat_vec(double[:,::1]A, double[::1]b, double[::1]c):
    """
    c = A * b
    :param A: n*m matrix
    :param b: m*1 vector
    :param c: n*1 vector
    :return:c
    """
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t m = A.shape[1]
    cdef Py_ssize_t i, j
    for i in xrange(n):
        c[i] = 0
        for j in xrange(m):
            c[i] += A[i,j] * b[j]
    return c


cdef class RandNoRep(object):
    """
    generate random numbers between 0 to n-1 without replacement
    """
    cdef Py_ssize_t [::1] seeds
    cdef Py_ssize_t n
    cdef Py_ssize_t b_size # batch size

    def __init__(self, int n):
        self.seeds = np.zeros(n, dtype=np.intp)
        self.n = n
        cdef Py_ssize_t i
        for i in range(n):
            self.seeds[i] = i

    cdef k_choice(self, Py_ssize_t [::1] arr, Py_ssize_t k):
        """
        return k elements w/o replacement, the amortized complexity is linear.
        """
        assert (k <= self.n-1)
        assert (k <=arr.size)
        cdef Py_ssize_t pt = self.n-1
        cdef Py_ssize_t i, j, tmp
        for i in range(k):
            j = rand() % self.n
            tmp = self.seeds[pt]
            self.seeds[pt] = self.seeds[j]
            self.seeds[j] = tmp
            arr[i] = self.seeds[pt]
            pt -= 1
