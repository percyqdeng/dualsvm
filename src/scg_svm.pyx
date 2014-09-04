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
cdef extern from "cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor=101
        CblasColMajor=102
    enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
        AtlasConj=114

    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY
                             ) nogil
    void dgemv "cblas_dgemv"(CBLAS_ORDER Order,
                      CBLAS_TRANSPOSE TransA, int M, int N,
                      double alpha, double *A, int lda,
                      double *X, int incX, double bgm,
                      double *Y, int incY) nogil
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def scg_md_svm(double[:,::1] ktr, int[::1] ytr, double[:,::1]kte=None, int[::1]yte=None,
                          double lmda=1E-5, int nsweep=1000, int b=5, int c=1):
    """
    stochastic coordinate descent on the dual svm, random sample a batch of data and update on another random sampled
    variables
    min 0.5*a'*kk*a - a'*1
     sub to: 0 <= a <= C
    where kk = 1/(lmda) * diag(y)*ktr*diag(y)
    :param lmda:
    :param nsweep:
    :param T:
    :param c, b  batchsize, c
    """
    cdef int n = ktr.shape[0]
    cdef int T = n * nsweep - 1
    cdef double C = 1.0/n
    cdef double [:,::1] kk = np.zeros((n, n))
    cdef unsigned int i,j,k
    cdef double start_time = time.time()
    cdef vector[int] nnzs
    cdef vector[double]err_tr
    cdef vector[int] num_opers
    cdef vector[double] obj
    cdef vector [double] obj_primal
    # cdef vector[double] snorm_grad
    cdef double [::1] alpha = np.zeros(n)  # the most recent solution
    cdef double [::1] a_tilde = np.zeros(n)  # the aCumulated solution in the path
    cdef double [::1] a_avg = np.zeros(n)
    cdef double [::1] kk_a = np.zeros(n)
    cdef vector[double] err_te
    # print"------------estimate parameters and set up variables----------------- "
    # comment:  seems very sensitive to the parameter estimation, if I use the second D_t, the algorithm diverges
    cdef double [::1] lip = np.zeros(n)
    cdef double l_max = 0
    for i in xrange(n):
        lip[i] = kk[i,i]
        if l_max < lip[i]:
            l_max = lip[i]
    Q = 1
    cdef double D_t = Q * np.sqrt(1.0/(2*n))
    cdef double res = 0
    cdef double gm
    cdef unsigned int[::1] batch_ind = np.zeros(c, dtype=np.uint32)
    cdef unsigned int var_ind
    cdef int[::1] uu = np.zeros(n, dtype=np.int32)
    cdef double stoc_cg
    # index of update, uu[i] = t+1 means the most recent update of
    # ith coordinate is in alpha[t+1], t = 0,1,...,T
    cdef int showtimes = 3
    cdef unsigned int t = 0
    cdef int count = 0 # count number of kernel products
    cdef double time_gen_rand = 0
    cdef int n_iters = 1 # stepsize to record the output, 1,2,4,8,...
    cdef unsigned int [::1] used = np.zeros(n, dtype=np.uint32)
    cdef unsigned int total_nnzs = 0

    start_time = time.time()
    for i in xrange(n):
        for j in xrange(n):
            kk[i,j] = ktr[i,j] * ytr[i] * ytr[j] / lmda
    sig_list = esti_std(np.asarray(kk), C, c)
    for i in xrange(n):
        res += sig_list[i]**2
    sig = np.sqrt(res)
    gm = 100*fmin(1.0/(2*l_max), 1.0/((1+T) * sig))
    # print "time to make new matrix %f " % (time.time()-start_time)
    print "estimated sigma: %f lipschitz: %f, gm: %e" % (sig, l_max, gm)
    # print "time for initialization %f" % (time.time()-start_time)
    # print "----------------------start the algorithm----------------------"
    start_time = time.time()
    for i in xrange(nsweep):
        for j in xrange(n):
            for k in xrange(c):
                batch_ind[k] = (rand() % n)
            var_ind = (rand() % n)
            # delta[t + 1] = delta[t] + thgm[t]
            stoc_cg = 0
            for k in xrange(c):
                stoc_cg += kk[var_ind,batch_ind[k]] * alpha[batch_ind[k]]
            stoc_cg *= float(n) / c
            stoc_cg -= 1
            a_tilde[var_ind] += ((t + 1) - uu[var_ind]) * alpha[var_ind]
            res = alpha[var_ind] - gm * stoc_cg
            if res > C:
                alpha[var_ind] = C
            elif res < 0:
                alpha[var_ind] = 0
            else:
                alpha[var_ind] = res
            # alpha[var_ind] = fmax(0, fmin(res, C))
            # alpha[var_ind] = cy_max(0, cy_min(res, C))
            if not used[var_ind]:
                used[var_ind] = 1
                total_nnzs += 1
            uu[var_ind] = t + 1
            count += (c * b)
            if t+1 == n_iters:
                n_iters *= 3
                # output the a_average
                # for j in xrange(n):
                #     a_avg[j] = (a_tilde[j] + ((t+2)-uu[j]) * alpha[j]) / (t+1)
                # mat_vec(kk, a_avg, kk_a)
                # res = 0
                # for j in xrange(n):
                #     res+= 0.5 * a_avg[j] * kk_a[j] - a_avg[j]
                # obj.push_back(-res)  # the dual of svm is the negative of our objective
                # res = 0
                # for j in xrange(n):
                #     res += fmax(0,1 - kk_a[j])/n + 0.5 * a_avg[j] * kk_a[j]

                mat_vec(kk,alpha,kk_a)
                res = 0
                for j in xrange(n):
                    res+= 0.5 * alpha[j] * kk_a[j] - alpha[j]
                obj.push_back(-res)
                res = 0
                for j in xrange(n):
                    res += fmax(0,1 - kk_a[j])/n + 0.5 * alpha[j] * kk_a[j]
                obj_primal.push_back(res)
                #--------------compute norm of gradient ---------------------
                # res = 0
                # for j in xrange(n):
                #     res += (kk_a[j]-1)**2
                # snorm_grad.push_back(res)
                #--------------compute train/test error rate
                res = 0
                for j in xrange(n):
                    res += (kk_a[j] <= 0)
                err_tr.push_back(res/n)
                if not(kte is None):
                    err = err_rate_test(yte, kte, ytr, a_avg)
                    err_te.push_back(err)
                #--------------4, compute number of kernel products
                num_opers.push_back(count)
                nnzs.push_back(total_nnzs)
            t += 1
    print "# of loops: %d, time of scd %f " % (nsweep*n, time.time()-start_time)
    for i in xrange(n):
        a_tilde[i] += (T + 2 - uu[i]) * alpha[i]
    for i in xrange(n):
        a_avg[i] = a_tilde[i] / (T + 1)
    return np.asarray(a_avg), err_tr, err_te, obj, obj_primal, num_opers, nnzs,


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def scg_da_svm(double[:,::1] ktr, int[::1] ytr, double[:,::1]kte=None, int[::1]yte=None, int verbose=True,
                          double lmda=1E-5, int nsweep=1000, double rho = 1.0, int c=1, int b=5):
    """
    stochastic coordinate gradient dual average on the dual svm, random sample a batch of data and update on another random sampled
    variables
    min 0.5*a'*kk*a - a'*1
     sub to: 0 <= a <= C
    where kk = 1/(lmda) * diag(y)*ktr*diag(y)
    :param ktr:
    :param ytr:
    :param kte:  kernel matrix of test data with train data
    :param yte:
    :param lmda:
    :param nsweep:
    :param T:
    :param c, b  batchsize, c
    """
    cdef int n = ktr.shape[0]
    cdef int T = n * nsweep - 1
    cdef double C = 1.0/n
    cdef double [:,::1] kk = np.zeros((n, n))
    # cdef np.ndarray[double, ndim=2, mode='c'] kk = np.zeros((n, n))
    cdef unsigned int i,j,k, Ii, Jj
    cdef double start_time = time.time()
    cdef vector[int] nnzs
    cdef vector[double]err_train
    cdef vector[int] num_opers
    cdef vector[double] obj
    cdef vector [double] obj_primal
    # cdef vector[double] snorm_grad
    cdef double [::1] alpha = np.zeros(n)  # the most recent solution
    cdef double [::1] a_tilde = np.zeros(n)  # the aCumulated solution in the path
    cdef double [::1] a_avg = np.zeros(n)
    cdef double [::1] kk_a = np.zeros(n)
    cdef double [::1] g_tilde = np.zeros(n)
    cdef vector[double] err_test
    # print"------------estimate parameters and set up variables----------------- "
    # comment:  seems very sensitive to the parameter estimation, if I use the second D_t, the algorithm diverges
    cdef double [::1] lip = np.zeros(n)
    cdef double l_max = b
    cdef Py_ssize_t interval
    interval = T / 20
    cdef Py_ssize_t scalar = 2
    cdef double res = 0
    cdef double gm
    cdef int[::1] uu = np.zeros(n, dtype=np.int32)
    cdef Py_ssize_t [::1] I = np.zeros(c, dtype=np.intp)
    cdef Py_ssize_t [::1] J = np.zeros(b, dtype=np.intp)
    cdef double stoc_cg, sig_D
    # index of update, uu[i] = t+1 means the most recent update of
    # ith coordinate is in alpha[t+1], t = 0,1,...,T
    cdef unsigned int t = 0
    cdef int count = 0 # count number of kernel products
    cdef double time_gen_rand = 0
    cdef Py_ssize_t n_iters = 1 # stepsize to record the output, 1,2,4,8,...
    cdef unsigned int [::1] used = np.zeros(n, dtype=np.uint32)
    cdef Py_ssize_t total_nnzs = 0
    for i in xrange(n):
        for j in xrange(n):
            kk[i,j] = ktr[i,j] * ytr[i] * ytr[j] / lmda
    cdef has_test = not(kte is None)
    sig_D = rho * n / lmda
    gm = fmax(2*l_max, np.sqrt(2*b*(T+1.0)/n)) * sig_D
    # gm = fmin(1.0/(2*l_max), 1.0/(np.sqrt(2*n*(1+T))*kappa))
    # print "estimated gamma: %f" % gm
    # print "time for initialization %f" % (time.time()-start_time)
    # print "----------------------start the algorithm----------------------"
    r1 = RandNoRep(n)
    r2 = RandNoRep(n)
    r1.k_choice(I, c)
    r2.k_choice(J, b)
    for t in xrange(T):
        r1.k_choice(I, c)
        r2.k_choice(J, b)
        # for i in range(c):
        #     I[i] = rand() % n
        # for j in range(b):
        #     J[j] = rand() %n
        for j in xrange(b):
            Jj = J[j]
            stoc_cg = 0
            for i in xrange(c):
                Ii = I[i]
                stoc_cg += kk[Jj, Ii] * alpha[Ii]
            stoc_cg *= float(n) / c
            stoc_cg -= 1
            g_tilde[Jj] += stoc_cg
            a_tilde[Jj] += ((t + 1) - uu[Jj]) * alpha[Jj]
            uu[Jj] = t+1
            alpha[Jj] = fmin(fmax(0, -g_tilde[Jj] / gm), C)
        count += b * c
        if t+1 == n_iters and verbose:
            mat_vec(kk,alpha,kk_a)
            res = 0
            for j in xrange(n):
                res+= 0.5 * alpha[j] * kk_a[j] - alpha[j]
            obj.push_back(res)
            res = 0
            for j in xrange(n):
                res += fmax(0,1 - kk_a[j])/n + 0.5 * alpha[j] * kk_a[j]
            obj_primal.push_back(res)
            #--------------compute train/test error rate
            res = 0
            for j in xrange(n):
                res += (kk_a[j] <= 0)
            err_train.push_back(res/n)
            if has_test:
                err = err_rate_test(yte, kte, ytr, alpha)
                err_test.push_back(err)
            #--------------4, compute number of kernel products
            num_opers.push_back(count)
            nnzs.push_back(total_nnzs)
            # n_iters += interval
            n_iters *= scalar
    # print "# of loops: %d, time of scd %f " % (nsweep*n, time.time()-start_time)
    for i in xrange(n):
        a_tilde[i] += (T + 2 - uu[i]) * alpha[i]
    for i in xrange(n):
        a_avg[i] = a_tilde[i] / (T + 1)
    if has_test:
        return np.asarray(a_avg), np.asarray(err_train), np.asarray(err_test), obj, obj_primal, num_opers, nnzs
    else:
        return np.asarray(a_avg), err_train, obj, obj_primal, num_opers, nnzs


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def cd_svm(double[:,::1] ktr, int[::1] ytr, double[:,::1] kte=None, int[::1] yte=None, int nsweep=100, double lmda=1E-5, int verbose=True):
    """
    coordinate descent for dual svm, Nesterov's algorithm
    param: nsweep, number of iteration in coordinate descent
    """
    cdef Py_ssize_t n = ktr.shape[0]
    cdef double C = 1.0/n
    cdef double [:,::1] kk = np.zeros((n, n))
    cdef double [::1] alpha = np.zeros(n)  # generated solutions
    cdef double[::1] z = np.zeros(n)
    cdef vector[double] obj_train
    cdef vector [double] err_train
    cdef vector [double] err_test
    cdef vector [double] num_opers
    cdef int T = nsweep
    cdef int has_test = not(kte is None)
    cdef unsigned int i,j,k, t
    cdef double start_time = time.time()
    cdef double A, B, res, new_alpha_i
    cdef Py_ssize_t interval, scalar=2, n_iters=1

    interval = np.maximum(1, T / 20)
    for i in xrange(n):
        for j in xrange(n):
            kk[i,j] = ktr[i,j] * ytr[i] * ytr[j] / lmda
    for t in xrange(T+1):
        i = (rand() % n)
        A = kk[i, i]
        B = z[i] - kk[i, i] * alpha[i] - 1
        res = -B / A
        new_alpha_i = fmax(0, fmin(res, C))
        for j in xrange(n):
            z[j] += (new_alpha_i - alpha[i]) * kk[i, j]
        alpha[i] = new_alpha_i
        if verbose and t == n_iters:
            num_opers.push_back((t+1)*n)
            res = 0
            for j in xrange(n):
                res += 0.5*alpha[j] * z[j] - alpha[j]
            obj_train.push_back(res)
            res = 0
            for j in xrange(n):
                res += (z[j] <= 0)
            err_train.push_back(res/n)
            if has_test:
                res = err_rate_test(yte, kte, ytr, alpha)
                err_test.push_back(res)
            # n_iters += interval
            n_iters *= scalar
    if has_test:
        return np.asarray(alpha), err_train, err_test, obj_train, num_opers
    else:
        return np.asarray(alpha), err_train, obj_train, num_opers



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
    """
    c = aa * b
    :param aa:
    :param b:
    :param c:
    :return:
    """
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


cdef esti_std(np.ndarray[double, ndim=2]kk, double C, int batchsize):
    n = kk.shape[0]
    # cdef double [:] sig = np.zeros(n)
    sig = np.zeros(n)
    # alpha = np.random.uniform(0,C, n)
    rep = 100
    for i in xrange(n):
        g = kk[i, :] * C
        sig[i] = np.std(g) * n / np.sqrt(batchsize)
    return sig

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

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef k_choice(self, Py_ssize_t [::1] arr, Py_ssize_t k):
        """
        return k elements w/o replacement, the amortized complexity is linear.
        """
        # assert (k <= self.n-1)
        # assert (k <=arr.size)
        cdef Py_ssize_t pt = self.n-1
        cdef Py_ssize_t i, j, tmp
        for i in range(k):
            j = rand() % self.n
            tmp = self.seeds[pt]
            self.seeds[pt] = self.seeds[j]
            self.seeds[j] = tmp
            arr[i] = self.seeds[pt]
            pt -= 1
