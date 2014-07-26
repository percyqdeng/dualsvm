# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
import time
from libcpp.vector cimport vector
from libc.stdlib cimport rand
from libcpp cimport bool
# cdef extern from "math.h"
from libc.math cimport fmax
from libc.math cimport fabs
from libc.math cimport fmin

ctypedef np.float64_t dtype_t
ctypedef np.int_t dtypei_t
# cimport rand_funcs


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)

def train_lassolr(double [:,::1] x, int[::1]y, int b=1, int c=2, double lmda=0.1, Py_ssize_t T=1000):
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
    assert(c<=p and b<=p)
    n = x.shape[0]
    p = x.shape[1]
    cdef Py_ssize_t [::1] I = np.zeros(c, dtype=np.intp)
    cdef Py_ssize_t [::1] J = np.zeros(b, dtype=np.intp)
    cdef Py_ssize_t [::1] u = np.zeros(p, dtype=np.intp)
    cdef double [::1] l = np.zeros(p)
    cdef int[::1] flag = np.zeros(p, dtype=np.intc)  # flag to check zero pattern, flag[i]=1 if w[i]=0; 0 otherwise
    cdef double[::1] w_tilde = np.zeros(p)
    cdef double[::1] w = np.zeros(p)
    cdef double[::1] w_bar = np.zeros(p)
    cdef double cur_grad
    cdef double[::1] accum_grad = np.zeros(p)  # accumulated stochastic coordinate gradient
    cdef Py_ssize_t ind
    cdef double sig, D, gm
    cdef vector [double] num_iter
    cdef vector [double] train_res
    cdef vector [double] num_zs # number of zeros
    cdef vector [double] sqnorm_grad
    cdef Py_ssize_t interval = 1
    cdef Py_ssize_t count
    cdef double tmp
    sig = np.sqrt(p**3/c)
    D = np.sqrt(p * 1)
    gm = fmax(2*c, np.sqrt(2.0*b*(T+1.0)/p)*sig/D)
    print "sigma %f, gamma %f" % (sig, gm)
    r1 = RandNoRep(p)
    r2 = RandNoRep(p)
    for t in xrange(T+1):
        ind = rand() % n
        r1.k_choice(I, c)
        r2.k_choice(J, b)
        tmp = 0
        for i in xrange(c):
            tmp += x[ind, I[i]] * w[I[i]]
        for j in xrange(b):
            cur_grad = tmp * x[ind, J[j]] * p / c
            cur_grad -= y[ind] * x[ind, J[j]]
            accum_grad[J[j]] += cur_grad
            w_tilde[J[j]] += (t+1 - u[J[j]]) * w[J[j]]
            l[J[j]] += lmda
            if fabs(accum_grad[J[j]]) <= l[J[j]]:
                w[J[j]] = 0
                flag[J[j]] = 1
            else:
                w[J[j]] = - (accum_grad[J[j]] - l[J[j]]*sign_func(accum_grad[J[j]]) ) / gm
                flag[J[j]] = 0
            u[J[j]] = t + 1
        if interval == t:
            # compute train error, # of zeros
            num_iter.push_back(t)
            train_res.push_back(eval_lasso_obj(w, x, y, lmda))
            interval *= 2
            # interval += 200
            count = 0
            for i in xrange(p):
               count += flag[i]
            num_zs.push_back(count)
    for j in xrange(p):
        w_bar[j] = (w_tilde[j] + (T+2 - u[j])*w[j]) / (T+1)
    return np.asarray(w_bar), train_res, num_zs, num_iter


def train_test_lassolr(double [:,::1] x, int[::1]y, double[:,::1]xtest, int[:]ytest,
                       int b=1, int c=2, double lmda=0.1, Py_ssize_t T=1000):
    """
    train and test lasso with limited resource, using stochastic coordinate gradient method
    :param x:
    :param y:
    :param b: batch size
    :param c: batch size
    :param lmda:
    :param T:
    :return: w_bar, train_res, test_res, num_zs, num_iter
    """
    cdef Py_ssize_t i, j, t
    cdef Py_ssize_t n, p
    assert(c<=p and b<=p)
    n = x.shape[0]
    p = x.shape[1]
    cdef Py_ssize_t [::1] I = np.zeros(c, dtype=np.intp)
    cdef Py_ssize_t [::1] J = np.zeros(b, dtype=np.intp)
    cdef Py_ssize_t [::1] u = np.zeros(p, dtype=np.intp)
    cdef double [::1] l = np.zeros(p)
    cdef int[::1] flag = np.zeros(p, dtype=np.intc)  # flag to check zero pattern, flag[i]=1 if w[i]=0; 0 otherwise
    cdef double[::1] w_tilde = np.zeros(p)
    cdef double[::1] w = np.zeros(p)
    cdef double[::1] w_bar = np.zeros(p)
    cdef double cur_grad
    cdef double[::1] accum_grad = np.zeros(p)  # accumulated stochastic coordinate gradient
    cdef Py_ssize_t ind
    cdef double sig, D, gm
    cdef vector [double] num_iter
    cdef vector [double] train_res
    cdef vector [double] test_res
    cdef vector [double] num_zs # number of zeros
    cdef vector [double] sqnorm_grad
    cdef Py_ssize_t interval = 1
    cdef Py_ssize_t count
    cdef double tmp
    sig = np.sqrt(p**3/c)
    D = np.sqrt(p * 1)
    gm = fmax(2*c, np.sqrt(2.0*b*(T+1.0)/p)*sig/D)
    print "sigma %f, gamma %f" % (sig, gm)
    r1 = RandNoRep(p)
    r2 = RandNoRep(p)
    for t in xrange(T+1):
        ind = rand() % n
        r1.k_choice(I, c)
        r2.k_choice(J, b)
        tmp = 0
        for i in xrange(c):
            tmp += x[ind, I[i]] * w[I[i]]
        for j in xrange(b):
            cur_grad = tmp * x[ind, J[j]] * p / c
            cur_grad -= y[ind] * x[ind, J[j]]
            accum_grad[J[j]] += cur_grad
            w_tilde[J[j]] += (t+1 - u[J[j]]) * w[J[j]]
            l[J[j]] += lmda
            if fabs(accum_grad[J[j]]) <= l[J[j]]:
                w[J[j]] = 0
                flag[J[j]] = 1
            else:
                w[J[j]] = - (accum_grad[J[j]] - l[J[j]]*sign_func(accum_grad[J[j]]) ) / gm
                flag[J[j]] = 0
            u[J[j]] = t + 1
        if interval == t:
            # compute train error, # of zeros
            num_iter.push_back(t)
            train_res.push_back(eval_lasso_obj(w, x, y, lmda))
            test_res.push_back(eval_lasso_obj(w, xtest, ytest, lmda))
            interval *= 2
            # interval += 200
            count = 0
            for i in xrange(p):
               count += flag[i]
            num_zs.push_back(count)
    for j in xrange(p):
        w_bar[j] = (w_tilde[j] + (T+2 - u[j])*w[j]) / (T+1)
    return np.asarray(w_bar), train_res, num_zs, num_iter


def train_lasso(double [:,::1] x, int[::1]y, double lmda=0.1, Py_ssize_t T=1000):
    """
    train lasso with full information, using dual average method
    :param x:
    :param y:
    :param lmda:
    :param T:
    :return:
    """

# def online_train_validate()
cdef inline sign_func(double x):
    if x >= 0:
        return 1
    else:
        return -1


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
