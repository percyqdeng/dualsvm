# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
import time
from libcpp.vector cimport vector
from libc.stdlib cimport rand
from libc.math cimport fmax
from libc.math cimport fabs
from libc.math cimport sqrt
from libc.math cimport fmin
import scipy.linalg.blas

ctypedef np.float64_t dtype_t
ctypedef np.int_t dtypei_t
# cimport rand_funcs
# cdef extern from "f2pyptr.h":
#     void *f2py_pointer(object) except NULL
#
# ctypedef double ddot_t(
#         int *N, double *X, int *incX, double *Y, int *incY)
# cdef ddot_t * ddot = <ddot_t*>f2py_pointer(scipy.linalg.blas.ddot._cpointer)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def train(double [:,::1] x, int[::1]y, double[:,::1]xtest=None, int[::1]ytest=None,  double lmda=0.1, double rho=0,  Py_ssize_t T=1000):
    """
    regularized dual averaging method on lasso
    :param x:
    :param y:
    :param b: batch size
    :param c: batch size
    :param lmda:
    :param rho: sparsity enhancing parameter
    :param T:
    :return: w_bar, train_res, num_zs, num_iter
    """
    cdef Py_ssize_t i, j, t
    cdef Py_ssize_t n, p
    n = x.shape[0]
    p = x.shape[1]
    cdef int[::1] flag = np.zeros(p, dtype=np.intc)  # flag to check zero pattern, flag[i]=1 if w[i]=0; 0 otherwise
    cdef double[::1] w = np.zeros(p)
    cdef double[::1] w_bar = np.zeros(p)
    cdef double[::1] g_bar = np.zeros(p)  # accumulated stochastic coordinate gradient
    cdef Py_ssize_t ind
    # L:  norm of gradient, D: sqrt of d(x), gm: = L/D
    cdef double L, D, gm
    cdef int has_test = not (xtest is None)
    cdef vector [double] num_iters
    cdef vector [double] num_features
    cdef vector [double] train_res
    cdef vector [double] test_res
    cdef vector [double] num_zs # number of zeros
    cdef vector [double] sqnorm_w
    cdef Py_ssize_t num_steps=1, interval = 100
    cdef Py_ssize_t count
    cdef double xw, lmda_t, res

    # estimate parameters
    D = sqrt(p)
    L = sqrt(p**3)
    gm = L / D
    for t in xrange(1, T):
        i = rand() % n
        xw = myddot(x[i], w)
        for j in xrange(p):
            g_bar[j] = (t-1.0)/t * g_bar[j] + 1.0/t * (x[i, j] * xw - y[i] * x[i, j])
            lmda_t = lmda + gm * rho / sqrt(t)
            w[j] = -sqrt(t)/gm * sign_func(g_bar[j])*fmax(fabs(g_bar[j])-lmda_t, 0)
            flag[j] = (fabs(g_bar[j]) <=lmda_t)
            w_bar[j] = (t-1.0)/t * w_bar[j] + 1.0/t * w[j]
        if t == num_steps:
            count = 0
            res = 0
            for j in xrange(p):
                count += flag[j]
                res += w[j] * w[j]
            sqnorm_w.push_back(res)
            num_zs.push_back(count)
            num_features.push_back(t * p)
            num_iters.push_back(t)
            train_res.push_back(eval_lasso_obj(w, x, y, lmda))
            if has_test:
                test_res.push_back(eval_lasso_obj(w, xtest, y, lmda))
            num_steps += interval
    if has_test:
        return np.asarray(w_bar), train_res, test_res, num_zs, num_iters, num_features, sqnorm_w
    else:
        return np.asarray(w_bar), train_res, num_zs, num_iters, num_features, sqnorm_w


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef myddot(double[::1]x, double[::1]y):
    cdef Py_ssize_t n = y.size
    cdef Py_ssize_t i
    cdef double res = 0
    for i in xrange(n):
        res += x[i] * y[i]

    return res


# def online_train_validate()
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
cdef eval_lasso_obj(double[::1]w, double[:,::1] x, int[::1]y, double lmda):
    """
    evaluate objective 0.5* ||Y-Xw||^2 + lmda * |w|
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
