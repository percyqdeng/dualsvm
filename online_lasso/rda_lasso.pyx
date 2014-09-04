# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
import time
from libcpp cimport bool
from cpython cimport bool
from libcpp.vector cimport vector
from libc.stdlib cimport rand
from libc.math cimport fmax
from libc.math cimport fabs
from libc.math cimport sqrt
from libc.math cimport fmin
import scipy.linalg.blas
cdef extern from "time.h" nogil:
    ctypedef Py_ssize_t clock_t
    cdef clock_t clock()
    # int clock()
    cdef enum:
        CLOCKS_PER_SEC

# cimport rand_funcs
# cdef extern from "f2pyptr.h":
#     void *f2py_pointer(object) except NULL
# ctypedef double ddot_t(
#         int *N, double *X, int *incX, double *Y, int *incY)
# cdef ddot_t * ddot = <ddot_t*>f2py_pointer(scipy.linalg.blas.ddot._cpointer)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def train(double [:,::1] x, double[::1]y, double[:,::1]xtest=None, double[::1]ytest=None,  double lmda=0.1,
          double sig_D=-1, double rho=0, int verbose=2, Py_ssize_t T=1000):
    """
    regularized dual averaging method on lasso
    :param x:
    :param y:
    :param lmda:
    :param sig_D:  sigma over D, which is a tuned parameter
    :param rho: sparsity enhancing parameter
    :param T:
    :param verbose:  verbosity level, 0, no output; 1, output the timecost; 2 output the intermediate result
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
    cdef vector [Py_ssize_t] num_iters
    cdef vector [Py_ssize_t] num_features
    cdef vector [double] train_res
    cdef vector [double] test_res
    cdef vector [Py_ssize_t] num_zs # number of zeros
    # cdef vector [double] sqnorm_w
    cdef vector [double] timecost
    cdef Py_ssize_t num_steps=1, interval
    interval = int(fmax(1, T/20))
    if verbose == 0:
        interval = T-1
    cdef Py_ssize_t count
    cdef double xw, lmda_t, res
    cdef Py_ssize_t start_t, end_t,
    cdef double cpu_t
    # estimate parameters
    if sig_D < 0:
        D = sqrt(p)
        L = sqrt(p**3)
        gm = L / D
    else:
        gm = sig_D
    start_t = clock()
    for t in xrange(1, T+1):
        i = rand() % n
        xw = myddot(x[i], w)
        lmda_t = lmda + gm * rho / sqrt(t)
        for j in xrange(p):
            w_bar[j] +=  w[j]
            g_bar[j] = (t-1.0)/t * g_bar[j] + 1.0/t * (x[i, j] * xw - y[i] * x[i, j])
            w[j] = -sqrt(t)/gm * sign_func(g_bar[j])*fmax(fabs(g_bar[j])-lmda_t, 0)
            flag[j] = (fabs(g_bar[j]) <=lmda_t)
        if t == num_steps and verbose==2:
            end_t = clock()
            cpu_t += <double>(end_t - start_t) / CLOCKS_PER_SEC
            timecost.push_back(cpu_t)
            count = 0
            res = 0
            for j in xrange(p):
                count += flag[j]
                res += w[j] * w[j]
            num_zs.push_back(count)
            num_features.push_back(t * p)
            num_iters.push_back(t)
            train_res.push_back(eval_lasso_obj(w, x, y, lmda))
            if has_test:
                test_res.push_back(eval_lasso_obj(w, xtest, ytest, lmda))
            num_steps += interval
            start_t = clock()
    if verbose == 1:
        end_t = clock()
        print "time cost %f" % (<double>(end_t - start_t) / CLOCKS_PER_SEC)

    for j in xrange(p):
        w_bar[j] /= T
    if has_test:
        return np.asarray(w_bar), train_res, test_res, num_zs, num_iters, num_features, timecost
    else:
        return np.asarray(w_bar), train_res, num_zs, num_iters, num_features, timecost


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def train2(double [:,::1] x, double[::1]y, double[:,::1]xtest=None, double[::1]ytest=None,  int b=4, int c=1, double lmda=0.1,
          double sig_D = -1, double rho=0, int verbose=2, Py_ssize_t T=1000):
    """
    regularized dual averaging method adapted to lasso with limited information
    we use a different unbiased estimator of stochastic gradient
    :param x:
    :param y:
    :param b: batch size
    :param c: batch size
    :param lmda:
    :param sig_D:  sigma over D
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
    cdef double[::1] curr_g = np.zeros(p)
    cdef Py_ssize_t ind
    # L:  norm of gradient, D: sqrt of d(x), gm: = L/D
    cdef double L, D, gm
    cdef int has_test = not (xtest is None)
    cdef vector [Py_ssize_t] num_iters
    cdef vector [Py_ssize_t] num_features
    cdef vector [double] train_res
    cdef vector [double] test_res
    cdef vector [Py_ssize_t] num_zs # number of zeros
    # cdef vector [double] sqnorm_w
    cdef vector [double] timecost
    cdef Py_ssize_t [::1] I = np.zeros(c, dtype=np.intp)
    cdef Py_ssize_t [::1] J = np.zeros(b, dtype=np.intp)
    cdef Py_ssize_t num_steps=1, interval
    interval = int(fmax(1, T/20))
    if not verbose:
        interval = T-1
    cdef Py_ssize_t count
    cdef double xw, lmda_t, res
    cdef Py_ssize_t start_t, end_t,
    cdef double cpu_t
    r1 = RandNoRep(p)
    r2 = RandNoRep(p)
    # estimate parameters
    if sig_D < 0:
        D = sqrt(p)
        L = sqrt(p**4)
        gm = L / D
    else:
        gm = sqrt(p) * sig_D
    start_t = clock()
    for t in xrange(1, T+1):
        ind = rand() % n
        r1.k_choice(I, c)
        r2.k_choice(J, b)
        xw = 0
        for i in xrange(c):
            xw += x[ind, I[i]] * w[I[i]]
        # xw = myddot(x[i], w)
        lmda_t = lmda + gm * rho / sqrt(t)
        for j in xrange(p):
            curr_g[j] = 0
            w_bar[j] +=  w[j]
        for j in xrange(b):
            curr_g[J[j]] = xw * x[ind, J[j]] * p * p / (b * c) - y[ind] * x[ind, J[j]] *p /b
        for j in xrange(p):
            g_bar[j] = (t-1.0)/t * g_bar[j] + 1.0/t * curr_g[j]
            w[j] = -sqrt(t)/gm * sign_func(g_bar[j])*fmax(fabs(g_bar[j])-lmda_t, 0)
            flag[j] = (fabs(g_bar[j]) <=lmda_t)
        if t == num_steps and verbose==2:
            end_t = clock()
            cpu_t += <double> (end_t - start_t) / CLOCKS_PER_SEC
            timecost.push_back(cpu_t)
            count = 0
            res = 0
            for j in xrange(p):
                count += flag[j]
                res += w[j] * w[j]
            num_zs.push_back(count)
            num_features.push_back(t * (b+c))
            num_iters.push_back(t)
            train_res.push_back(eval_lasso_obj(w, x, y, lmda))
            if has_test:
                test_res.push_back(eval_lasso_obj(w, xtest, ytest, lmda))
            num_steps += interval
            start_t = clock()
    if verbose == 1:
        print 'time cost %f ' % (<double>(clock()-start_t) / CLOCKS_PER_SEC)
    for j in xrange(p):
        w_bar[j] /= T
    if has_test:
        return np.asarray(w_bar), train_res, test_res, num_zs, num_iters, num_features, timecost
    else:
        return np.asarray(w_bar), train_res, num_zs, num_iters, num_features, timecost

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef myddot(double[::1]x, double[::1]y):
    cdef Py_ssize_t n = x.size
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
