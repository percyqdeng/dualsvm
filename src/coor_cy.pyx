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

# from libc.time cimport clock()
ctypedef np.float64_t dtype_t
ctypedef np.int_t dtypei_t
# ctypedef int dtypei_t

# cdef extern from "stdlib.h":
#     double drand48()

# @cython.boundscheck(False)
# @cython.cdivision(True)
# @cython.wraparound(False)
def scgd_cy(double[:,::1] ktr, int[::1] ytr,
                          double[:,::1]kte=None, int[::1]yte=None,
                          double lmda=1E-5, int nsweep=1000, int batchsize=5):
    """
    stochastic coordinate descent on the dual svm, random sample a batch of data and update on another random sampled
    variables
    min 0.5*a'*kk*a - a'*1
     sub to: 0 <= a <= cc
    where kk = 1/(lmda) * diag(y)*ktr*diag(y)
    :param ktr:
    :param ytr:
    :param kte:
    :param yte:
    :param lmda:
    :param nsweep:
    :param T:
    :param batchsize:

    """
    cdef int n = ktr.shape[0]
    cdef int T = n * nsweep - 1
    cdef double cc = 1.0/n
    cdef double [:,::1] kk = np.zeros((n, n))
    cdef unsigned int i,j,k
    cdef double start_time = time.time()
    for i in xrange(n):
        for j in xrange(n):
            kk[i,j] = ktr[i,j] * ytr[i] * ytr[j] / lmda
    # print "time to make new matrix %f " % (time.time()-start_time)
    start_time = time.time()
    cdef vector[int] nnzs
    cdef vector[double]err_tr
    cdef vector[int] num_oper
    cdef vector[double] obj
    cdef vector [double] obj_primal
    cdef vector[double] snorm_grad
    cdef double err_count
    # cdef np.ndarray[double] pred = np.zeros(kte.shape[0])
    # cdef bool has_kte = True
    cdef vector[double] err_te
    # print"------------estimate parameters and set up variables----------------- "
    # comment:  seems very sensitive to the parameter estimation, if I use the second D_t, the algorithm diverges
    #
    cdef double [::1] lip = np.zeros(n)
    cdef double l_max = 0
    for i in xrange(n):
        lip[i] = kk[i,i]
        if l_max < lip[i]:
            l_max = lip[i]
    Q = 1
    cdef double D_t = Q * np.sqrt(1.0/(2*n))
    sig_list = esti_std(np.asarray(kk), cc, batchsize)
    cdef double res = 0
    for i in xrange(n):
        res += sig_list[i]**2
    sig = np.sqrt(res)
    cdef double [::1] eta = np.ones(T+1)
    cdef double [::1] theta = np.zeros(T+1)
    eta[0] = fmin(1.0 / (2 * l_max), D_t / sig * np.sqrt(float(n) / (1 + T)))
    for i in xrange(1, T+1):
        eta[i] = eta[0]
    for i in xrange(T+1):
        theta[i] = eta[i]
    cdef double [::1] alpha = np.zeros(n)  # the most recent solution
    cdef double [::1] a_tilde = np.zeros(n)  # the accumulated solution in the path
    cdef double [::1] a_avg = np.zeros(n)
    cdef double [::1] delta = np.zeros(T + 2)
    cdef double [::1] kk_a = np.zeros(n)
    cdef unsigned int[::1] batch_ind = np.zeros(batchsize, dtype=np.uint32)
    cdef unsigned int var_ind
    cdef int[::1] uu = np.zeros(n, dtype=np.int32)
    cdef double stoc_cg
    # index of update, uu[i] = t means the most recent update of
    # ith coordinate is in the t-th round, t = 0,1,...,T
    cdef int showtimes = 3
    cdef unsigned int t = 0
    cdef int count = 0 # count number of kernel products
    cdef double time_gen_rand = 0
    cdef int rec_step = 1 # stepsize to record the output, 1,2,4,8,...
    cdef unsigned int [::1] used = np.zeros(n, dtype=np.uint32)
    cdef unsigned int total_nnzs = 0
    cdef unsigned int [::1]ind_list = np.zeros(n, dtype=np.uint32)
    print "estimated sigma: %f lipschitz: %f, eta: %e" % (sig, l_max, eta[0])
    # print "time for initialization %f" % (time.time()-start_time)
    # print "----------------------start the algorithm----------------------"
    start_time = time.time()
    cdef unsigned int tmp
    for i in xrange(nsweep):
        # rand_perm(ind_list)
        for j in xrange(n):
            for k in xrange(batchsize):
                batch_ind[k] = (rand() % n)
            var_ind = (rand() % n)
            delta[t + 1] = delta[t] + theta[t]
            stoc_cg = 0
            for k in xrange(batchsize):
                stoc_cg += kk[var_ind,batch_ind[k]] * alpha[batch_ind[k]]
            stoc_cg *= float(n)/batchsize
            stoc_cg -= 1
            a_tilde[var_ind] += (delta[t + 1] - delta[uu[var_ind]]) * alpha[var_ind]
            res = alpha[var_ind] - eta[t] * stoc_cg
            if res > cc:
                alpha[var_ind] = cc
            elif res < 0:
                alpha[var_ind] = 0
            else:
                alpha[var_ind] = res
            # alpha[var_ind] = fmax(0, fmin(res, cc))
            # alpha[var_ind] = cy_max(0, cy_min(res, cc))
            if not used[var_ind]:
                used[var_ind] = 1
                total_nnzs += 1
            uu[var_ind] = t + 1
            count += batchsize
            if t+1 == rec_step:
                rec_step *= 3
                for j in xrange(n):
                    a_avg[j] = (a_tilde[j] + (delta[t+1]-delta[uu[j]]) * alpha[j]) / delta[t+1]
                mat_vec(kk, a_avg, kk_a)
                #--------------compute dual/primal objective of svm
                res = 0
                for j in xrange(n):
                    res+= 0.5 * a_avg[j] * kk_a[j] - a_avg[j]
                obj.push_back(-res)  # the dual of svm is the negative of our objective
                res = 0
                for j in xrange(n):
                    res += fmax(0,1 - kk_a[j])/n + 0.5 * a_avg[j] * kk_a[j]
                obj_primal.push_back(res)
                #--------------compute norm of gradient ---------------------
                res = 0
                for j in xrange(n):
                    res += (kk_a[j]-1)**2
                snorm_grad.push_back(res)
                #--------------compute train/test error rate
                res = 0
                for j in xrange(n):
                    res += (kk_a[j] <= 0)
                err_tr.push_back(res/n)
                if not(kte is None):
                    err = err_rate_test(yte, kte, ytr, a_avg)
                    err_te.push_back(err)
                #--------------4, compute number of kernel products
                num_oper.push_back(count)
                nnzs.push_back(total_nnzs)
            t += 1
        # if i % (nsweep / showtimes) == 0:
        #     print "# of sweeps " + str(i)
    print "# of loops: %d, time of scd %f " % (nsweep*n, time.time()-start_time)
    for i in xrange(n):
        a_tilde[i] += (delta[T + 1] - delta[uu[i]]) * alpha[i]
    for i in xrange(n):
        a_avg[i] = a_tilde[i] / delta[T + 1]
    return np.asarray(a_avg), err_tr, err_te, obj, obj_primal, num_oper, nnzs, snorm_grad


# cdef rand_perm(int[:] ind):
#     cdef int i, j, n = ind.shape[0]
#     cdef int tmp
#     for i in range(n):
#         j =i + int(n-i) * drand48()
#         if j >= n:
#             j = n -1
#         tmp = ind[i]
#         ind[i] = ind[j]
#         ind[j] = tmp


# cdef inline double cy_max(double a,double b):
#     return a if a >= b else b
#
#
# cdef inline double cy_min(double a, double b):
#     return a if a<b else b


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double err_rate_test(int[:]label, double[:,::1]k, int[:]y, double[:]a):
    cdef int n = k.shape[0]
    cdef int m = k.shape[1]
    cdef int i, j
    cdef double res, err = 0
    for i in xrange(n):
        res = 0
        for j in xrange(m):
            res += k[i,j] * y[j] *a[j]
        err += (label[i] * res<=0)
    return err / n

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef mat_vec(double[:,::1]aa, double[::1]b, double[::1]c):
    cdef int n = aa.shape[0]
    cdef int m = aa.shape[1]
    # cdef double [::1] c = np.zeros(n)
    assert (m == n)
    cdef int i, j
    for i in xrange(n):
        c[i] = 0
        for j in xrange(m):
            c[i] += aa[i,j] * b[j]
    return c

cdef esti_std(np.ndarray[double, ndim=2]kk, double cc, int batchsize):
    n = kk.shape[0]
    # cdef double [:] sig = np.zeros(n)
    sig = np.zeros(n)
    # alpha = np.random.uniform(0,cc, n)
    rep = 100
    for i in xrange(n):
        g = kk[i, :] * cc
        sig[i] = np.std(g) * n / np.sqrt(batchsize)
    return sig
