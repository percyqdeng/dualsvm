# distutils: language = c++

__author__ = 'qdengpercy'

cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
# from libcpp import bool
# from cpython cimport bool
from libc.stdlib cimport rand
# cdef extern from "math.h"
#     double fmax(double x, double y)
from libc.math cimport fmax
from libc.math cimport fmin
ctypedef np.float64_t dtype_t
ctypedef np.int_t dtypei_t
# ctypedef int dtypei_t



@cython.boundscheck(False)
@cython.cdivision(True)
cpdef stoch_coor_descent_cy(np.ndarray[double, ndim=2]ktr, ytr,
                          kte,  yte,
                          double lmda, int nsweep, int T, int batchsize):
    """
    stochastic coordinate descent on the dual svm, random sample a batch of data and update on another random sampled
    variables
    min 0.5*a'*kk*a - a'*b
    sub to: 0 <= a <= c
    kk = 1/lmda * diag(y)*ktr*diag(y)
    """
    cdef Py_ssize_t n = ktr.shape[0]
    cdef double cc = 1.0/n
    cdef np.ndarray[double, ndim=2] kk = np.zeros([n, n])
    cdef Py_ssize_t i,j,k
    for i in range(n):
        for j in range(n):
            kk[i,j] = ktr[i,j] * ytr[i] * ytr[j] / lmda

    cdef vector[int] nnz
    cdef vector[double]err_tr
    cdef vector[int] ker_oper
    cdef vector[double] obj
    cdef vector [double] obj_primal
    cdef double err_count
    cdef np.ndarray[double, ndim=1] pred = np.zeros(kte.shape[0])
    # cdef bool has_kte = True
    cdef vector[double] err_te
    print"------------estimate parameters and set up variables----------------- "
    # comment:  seems very sensitive to the parameter estimation, if I use the second D_t, the algorithm diverges
    #
    cdef np.ndarray[double] lip = np.zeros(n)
    cdef double l_max = 0
    for i in range(n):
        lip[i] = kk[i,i]
        if l_max < lip[i]:
            l_max = lip[i]
    Q = 1
    cdef double D_t = Q * np.sqrt(1.0/(2*n))
    # D_t = Q * (n / 2) * cc
    sig_list = esti_std(kk, cc, batchsize)
    cdef res = 0
    for i in range(n):
        res += sig_list[i]**2
    sig = np.sqrt(res)
    cdef np.ndarray[double] eta = np.ones(T+1)
    cdef np.ndarray[double] theta = np.zeros(T+1)
    eta[0] = np.minimum(1.0 / (2 * l_max), D_t / sig * np.sqrt(float(n) / (1 + T)))
    for i in range(1, T+1):
        eta[i] = eta[0]
    for i in range(T+1):
        theta[i] = eta[i]
    cdef np.ndarray[double] alpha = np.zeros(n)  # the most recent solution
    cdef np.ndarray[double]a_tilde = np.zeros(n)  # the accumulated solution in the path
    cdef np.ndarray[double] a_avg = np.zeros(n)
    cdef np.ndarray[double] delta = np.zeros(T + 2)
    cdef np.ndarray[double] kk_a = np.zeros(n)
    cdef np.ndarray[Py_ssize_t,ndim=1] samp_ind = np.zeros(batchsize, dtype=int)
    cdef Py_ssize_t var_ind
    cdef np.ndarray[Py_ssize_t] uu = np.zeros(n, dtype=int)
    cdef double stoc_coor_grad
    # index of update, u[i] = t means the most recent update of
    # ith coordinate is in the t-th round, t = 0,1,...,T
    cdef int showtimes = 5
    cdef Py_ssize_t t = 0
    cdef int count = 0

    print "estimated sigma: "+str(sig)+" lipschitz: "+str(l_max)
    print "----------------------start the algorithm----------------------"
    for i in range(nsweep):
        for j in range(n):
            for k in range(batchsize):
                samp_ind[k] = int(rand() % n)
            var_ind = int(rand() % n)
            delta[t + 1] = delta[t] + theta[t]
            stoc_coor_grad = 0
            for k in range(batchsize):
                stoc_coor_grad += kk[var_ind, samp_ind[k]] * alpha[samp_ind[k]] * n /batchsize - 1
            a_tilde[var_ind] += (delta[t + 1] - delta[uu[var_ind]]) * alpha[var_ind]
            res =  alpha[var_ind] - eta[t] * stoc_coor_grad
            alpha[var_ind] = fmax(0, fmin(res, cc))
            uu[var_ind] = t + 1
            t += 1
            count += batchsize
        if i % (nsweep / showtimes) == 0:
            print "# of sweeps " + str(i)
        #-------------compute the result after the ith sweep----------------
        if i % n == 0:
            for j in range(n):
                a_avg[j] = (a_tilde[j] + (delta[t]-delta[uu[j]]) * alpha[j]) / delta[t]
            kk_a = mat_vec(kk, a_avg)
            res = 0
            # 1, compute dual objective of svm
            for j in range(n):
                res+= 0.5 * a_avg[j] * kk_a[j] - a_avg[j]
            obj.push_back(res)
            err_count = 0
            for j in range(n):
                err_count += (kk_a[j] < 0)
            err_tr.push_back(float(err_count)/n)
            # 2, compute primal objective of svm
            res = 0
            for j in range(n):
                res += fmax(0,1 - lmda * kk_a[j])/n + 0.5 * a_avg[j] * kk_a[j]
            obj_primal.push_back(res)
            ker_oper.push_back(count)
            if True:
                # pred =
                pred = np.sign(np.dot(kte, ytr*a_avg))
                err = np.mean(yte*pred < 0)
                # err = cmp_err_rate(yte, pred)
                err_te.push_back(err)
    # -------------compute the final result after nsweep-th sweeps---------------
    a_tilde += (delta[T + 1] - delta[uu]) * alpha
    for i in range(n):
        alpha[i] = a_tilde[i] / delta[T + 1]
    return err_tr, err_te, obj, obj_primal, ker_oper


cdef cmp_err_rate(y, np.ndarray[double, ndim=1]z):
    cdef Py_ssize_t i, n
    n = len(y)
    cdef int err = 0
    for i in range(n):
        err += (y[i] * z[i] < 0)
    return float(err) / n

@cython.boundscheck(False)
@cython.cdivision(True)
cdef mat_vec(np.ndarray[double, ndim=2]aa, np.ndarray[double, ndim=1]b):
    cdef Py_ssize_t n = aa.shape[0]
    cdef Py_ssize_t m = aa.shape[1]
    cdef np.ndarray[double] c = np.zeros(n)
    assert (m == b.shape[0])
    cdef Py_ssize_t i, j
    for i in range(n):
        for j in range(m):
            c[i] += aa[i,j] * b[j]
    return c

cdef esti_std(np.ndarray[double, ndim=2]kk, double cc, int batchsize):
    n = kk.shape[0]
    # cdef np.ndarray[double] sig = np.zeros(n)
    sig = np.zeros(n)
    alpha = np.random.uniform(0,cc, n)
    rep = 100
    for i in range(n):
        g = kk[i, :] * cc
        sig[i] = np.std(g) * n / np.sqrt(batchsize)
    return sig