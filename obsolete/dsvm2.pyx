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

from auxiliary.rand_no_repeat cimport RandNoRep
from kernel_func cimport KernelFunc
from rand_dataset cimport

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
def stoc_mirror_descent(double[:,::1] ktr, double[::1] ytr, double[:,::1]kte=None, double[::1]yte=None,
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
    # cdef double [:,::1] kk = np.zeros((n, n))
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
        lip[i] = ktr[i,i]
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
    # for i in xrange(n):
    #     for j in xrange(n):
    #         kk[i,j] = ktr[i,j] * ytr[i] * ytr[j] / lmda
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
                stoc_cg += ytr[batch_ind[k]] * ktr[var_ind,batch_ind[k]] * alpha[batch_ind[k]]
            stoc_cg *= float(n) / c * ytr[var_ind]
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
                # res = 0
                # for j in xrange(n):
                #     res += fmax(0,1 - kk_a[j])/n + 0.5 * alpha[j] * kk_a[j]
                # obj_primal.push_back(res)
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
    return np.asarray(a_avg), err_tr, err_te, obj, num_opers,


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def stoc_dual_averaging(double[:,::1] ktr, double[::1] ytr, double[:,::1]kte=None, double[::1]yte=None, int verbose=True,
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
    cdef Py_ssize_t i,j,k, t, n, ntest, p, T, Ii, Jj
    cdef Py_ssize_t count, interval, scalar, n_iters, total_nnzs
    cdef double C, start_time, l_max, gm, res, stoc_cg, sig_D, time_gen_rand
    cdef int has_test = xte is not None
    n = xtr.shape[0]
    p = xtr.shape[1]
    # start_time = time.time()
    cdef vector[int] nnzs, num_opers
    cdef vector[double]err_train, obj, obj_primal, err_test
    cdef double [::1] alpha = np.zeros(n)  # the most recent solution
    cdef double [::1] a_tilde = np.zeros(n)  # the accumulated solution in the path
    cdef double [::1] a_avg = np.zeros(n)
    cdef double [::1] z = np.zeros(n)
    cdef double [::1] z2
    cdef double [::1] g_tilde = np.zeros(n)
    cdef Py_ssize_t [::1] I = np.zeros(c, dtype=np.intp)
    cdef Py_ssize_t [::1] J = np.zeros(b, dtype=np.intp)
    # index of update, uu[i] = t+1 means the most recent update of
    cdef int[::1] uu = np.zeros(n, dtype=np.int32)
    cdef unsigned int [::1] used = np.zeros(n, dtype=np.uint32)
    # print"------------estimate parameters and set up variables----------------- "
    cdef double [::1] lip = np.zeros(n)
    # print"------------estimate parameters and set up variables----------------- "
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
    # for i in xrange(n):
    #     for j in xrange(n):
    #         kk[i,j] = ktr[i,j] * ytr[i] * ytr[j] / lmda
    cdef int has_test = not(kte is None)
    sig_D = n / lmda
    gm = rho * fmax(2*l_max, np.sqrt(2*b*(T+1.0)/n)) * sig_D
    # gm = fmin(1.0/(2*l_max), 1.0/(np.sqrt(2*n*(1+T))*kappa))
    # print "estimated gamma: %f" % gm
    # print "time for initialization %f" % (time.time()-start_time)
    # ----------------------start the algorithm----------------------
    cdef RandNoRep r1 = RandNoRep(n)
    cdef RandNoRep r2 = RandNoRep(n)
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
                stoc_cg += ktr[Jj, Ii] * alpha[Ii] * ytr[Ii]
            stoc_cg *= ytr[Jj] * float(n) / c
            stoc_cg -= 1
            g_tilde[Jj] += stoc_cg
            a_tilde[Jj] += ((t + 1) - uu[Jj]) * alpha[Jj]
            uu[Jj] = t+1
            alpha[Jj] = fmin(fmax(0, -g_tilde[Jj] / gm), C)
        count += b * c
        if t+1 == n_iters and verbose:
            k_dot_yalpha(ytr, alpha, z, ktr)
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
    return np.asarray(a_avg), np.asarray(err_train), np.asarray(err_test), np.asarray(obj), np.asarray(num_opers)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def stocda_on_fly(double[:,::1] xtr, double[::1] ytr, KernelFunc kernel, double[:,::1]xte=None, double[::1]yte=None,
                      int verbose=True, double lmda=1E-5, int nsweep=1000, double rho = 1.0, int c=1, int b=5):
    """
    stochastic coordinate gradient dual average on the dual svm,
    random sample a batch of data and update on another random sampled variables
    kernel product is computed on the fly
    min 0.5*a'*kk*a - a'*1
     sub to: 0 <= a <= C
    where kk = 1/(lmda) * diag(y)*ktr*diag(y)
    :param c, b  batchsize, c
    """
    cdef Py_ssize_t i,j,k, t, n, ntest, p, T, Ii, Jj
    cdef Py_ssize_t count, interval, scalar, n_iters, total_nnzs
    cdef double C, start_time, l_max, gm, res, stoc_cg, sig_D, time_gen_rand
    cdef int has_test = xte is not None
    n = xtr.shape[0]
    p = xtr.shape[1]
    # start_time = time.time()
    cdef vector[int] nnzs, num_opers
    cdef vector[double]err_train, obj, obj_primal, err_test
    cdef double [::1] alpha = np.zeros(n)  # the most recent solution
    cdef double [::1] a_tilde = np.zeros(n)  # the accumulated solution in the path
    cdef double [::1] a_avg = np.zeros(n)
    cdef double [::1] z = np.zeros(n)
    cdef double [::1] z2
    cdef double [::1] g_tilde = np.zeros(n)
    cdef Py_ssize_t [::1] I = np.zeros(c, dtype=np.intp)
    cdef Py_ssize_t [::1] J = np.zeros(b, dtype=np.intp)
    # index of update, uu[i] = t+1 means the most recent update of
    cdef int[::1] uu = np.zeros(n, dtype=np.int32)
    cdef unsigned int [::1] used = np.zeros(n, dtype=np.uint32)
    # print"------------estimate parameters and set up variables----------------- "
    cdef double [::1] lip = np.zeros(n)

    #---------------- initialize value
    if has_test:
        ntest = xte.shape[0]
        z2 = np.zeros(ntest)
    l_max = b
    T = n * nsweep - 1
    interval = T / 20
    scalar = 2
    res = 0
    T = n * nsweep - 1
    C = 1.0/n
    count = 0 # count number of kernel products
    time_gen_rand = 0
    n_iters = 1 # stepsize to record the output, 1,2,4,8,...
    total_nnzs = 0
    sig_D = n / lmda
    gm = rho * fmax(2*l_max, np.sqrt(2*b*(T+1.0)/n)) * sig_D
    # gm = fmin(1.0/(2*l_max), 1.0/(np.sqrt(2*n*(1+T))*kappa))
    # print "time for initialization %f" % (time.time()-start_time)
    # "----------------------start the algorithm----------------------"
    cdef RandNoRep r1 = RandNoRep(n)
    cdef RandNoRep r2 = RandNoRep(n)
    r1.k_choice(I, c)
    r2.k_choice(J, b)
    for t in xrange(T):
        r1.k_choice(I, c)
        r2.k_choice(J, b)
        for j in xrange(b):
            Jj = J[j]
            stoc_cg = 0
            for i in xrange(c):
                Ii = I[i]
                stoc_cg += kernel.product(xtr[Jj], xtr[Ii]) * ytr[Ii] *alpha[Ii]
            stoc_cg *= float(n) / (lmda * c) * ytr[Jj]
            stoc_cg -= 1
            g_tilde[Jj] += stoc_cg
            a_tilde[Jj] += ((t + 1) - uu[Jj]) * alpha[Jj]
            uu[Jj] = t+1
            alpha[Jj] = fmin(fmax(0, -g_tilde[Jj] / gm), C)
        count += b * c
        if t+1 == n_iters and verbose:
            k_dot_yalpha(ytr, alpha, z, xtr, xtr, kernel)
            res = zero_one_loss(ytr, z)
            err_train.push_back(res)
            res = eval_dsvm_obj(z, ytr, alpha, lmda)
            obj.push_back(res)
            if has_test:
                k_dot_yalpha(ytr, alpha, z2,xte, xtr, kernel)
                err = zero_one_loss(yte, z2)
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
    return np.asarray(a_avg), np.asarray(err_train), np.asarray(err_test), np.asarray(obj), np.asarray(num_opers),

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def coord_descent(double[:,::1] ktr, double[::1] ytr, double[:,::1] kte=None, double[::1] yte=None, int nsweep=100, double lmda=1E-5, int verbose=True):
    """
    coordinate descent for dual svm, Nesterov's algorithm
    param: nsweep, number of iteration in coordinate descent
    z = ktr.dot(y*alpha)
    """
    cdef Py_ssize_t n = ktr.shape[0]
    cdef double C = 1.0/n
    # cdef double [:,::1] kk = np.zeros((n, n))
    cdef double [::1] alpha = np.zeros(n)  # generated solutions
    cdef double[::1] z = np.zeros(n)
    cdef double [::1] z2
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
    if kte is not None:
        z2 = np.zeros(kte.shape[0])
    # for i in xrange(n):
    #     for j in xrange(n):
    #         kk[i,j] = ktr[i,j] * ytr[i] * ytr[j] / lmda
    for t in xrange(T+1):
        i = (rand() % n)
        A = ktr[i, i] / lmda
        B = (ytr[i] * z[i] - ktr[i, i] * alpha[i])/lmda - 1
        res = -B / A
        new_alpha_i = fmax(0, fmin(res, C))
        for j in xrange(n):
            z[j] += (new_alpha_i - alpha[i]) * ktr[i, j] * ytr[i]
        alpha[i] = new_alpha_i
        if verbose and t == n_iters:
            num_opers.push_back((t+1)*n)
            res = 0
            res = eval_dsvm_obj(z, ytr, alpha, lmda)
            # for j in xrange(n):
            #     res += 0.5*alpha[j] * z[j] - alpha[j]
            obj_train.push_back(res)
            res = zero_one_loss(ytr, z)
            # for j in xrange(n):
            #     res += (z[j] <= 0)
            err_train.push_back(res/n)
            if has_test:
                k_dot_yalpha(ytr, alpha, z2, kte)
                res = zero_one_loss(yte, z2)
                # res = err_rate_test(yte, kte, ytr, alpha)
                err_test.push_back(res)
            # n_iters += interval
            n_iters *= scalar
    return np.asarray(alpha), np.asarray(err_train), np.asarray(err_test), np.asarray(obj_train), np.asarray(num_opers)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def cd_on_the_fly(double[:,::1] xtr, double[::1] ytr, KernelFunc kernel, double[:,::1] xte=None, double[::1] yte=None,
                      int nsweep=100, double lmda=1E-5, int verbose=True):
    """
    coordinate descent for dual svm, Nesterov's algorithm
    param: nsweep, number of iteration in coordinate descent
    """
    cdef Py_ssize_t n, n_test, T, i, j, k, t, interval, scalar=2, n_iters
    cdef int has_test
    n = xtr.shape[0]
    cdef double C, A, B, kii, res, new_alpha_i, start_time
    # cdef double [:,::1] kk = np.zeros((n, n))
    cdef double [::1] alpha = np.zeros(n)  # generated solutions
    cdef double[::1] z = np.zeros(n)
    cdef double[::1] z2
    cdef vector[double] obj_train
    cdef vector [double] err_train
    cdef vector [double] err_test
    cdef vector [double] num_opers
    T = nsweep
    has_test = not(xte is None)
    start_time = time.time()

    #-----initialize value----------
    C = 1.0/n
    scalar=2
    n_iters=1
    if has_test:
        n_test = xte.shape[0]
        z2 = np.zeros(n_test)

    interval = np.maximum(1, T / 20)
    for t in xrange(T+1):
        i = (rand() % n)
        kii = kernel.product(xtr[i], xtr[i])
        A = kii / lmda
        B = (ytr[i] * z[i] - kii * alpha[i])/lmda - 1
        res = -B / A
        new_alpha_i = fmax(0, fmin(res, C))
        for j in xrange(n):
            z[j] += ytr[i] * (new_alpha_i - alpha[i]) * kernel.product(xtr[i], xtr[j])
        alpha[i] = new_alpha_i
        if verbose and t == n_iters:
            num_opers.push_back((t+1)*n)
            res = eval_dsvm_obj(z, ytr, alpha, lmda)
            obj_train.push_back(res)
            res = zero_one_loss(ytr, z)
            err_train.push_back(res)
            if has_test:
                k_dot_yalpha(ytr, alpha, z2, xte, xtr, kernel)
                res = zero_one_loss(yte, z2)
                err_test.push_back(res)
            # n_iters += interval
            n_iters *= scalar
    return np.asarray(alpha), np.asarray(err_train), np.asarray(err_test), np.asarray(obj_train), np.asarray(num_opers)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double zero_one_loss(double[::1] label, double[::1] pred):
    cdef Py_ssize_t i, n
    n = label.shape[0]
    assert (n == pred.shape[0])
    cdef double err = 0
    for i in xrange(n):
        err += (label[i] * pred[i] < 0)
    return err / n


# @cython.boundscheck(False)
# @cython.cdivision(True)
# @cython.wraparound(False)
# cdef double err_rate_test(double[:]label, double[:,::1]k, double[:]y, double[:]a):
#     cdef int n = k.shape[0]
#     cdef int m = k.shape[1]
#     cdef int i, j
#     cdef double res, err = 0
#     for i in xrange(n):
#         res = 0
#         for j in xrange(m):
#             res += k[i,j] * y[j] *a[j]
#         err += (label[i] * res<=0)
#     return err / n


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void k_dot_yalpha( double[::1]y, double[::1]alpha, double [::1]z, double[:,::1]xtest, double[:, ::1]x=None,
                        KernelFunc kernel=None):
    """
    the classifier output on test data
    z <- k.dot(y*alpha)
    """
    cdef Py_ssize_t n, m, i, j
    if kernel is None:
        # xtest is kte
        n = xtest.shape[0]
        m = xtest.shape[1]
        for i in xrange(n):
            z[i] = 0
            for j in xrange(m):
                z[i] += xtest[i, j] * y[j] * alpha[j]
    else:
        n = xtest.shape[0]
        m = x.shape[0]
        for i in xrange(n):
            z[i] = 0
            for j in xrange(m):
                z[i] += kernel.product(xtest[i], x[j]) * y[j] * alpha[j]


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double eval_dsvm_obj(double [::1]z, double [::1]y, double[::1] alpha, double lmda):
    cdef Py_ssize_t i, j, n, p
    n = z.shape[0]
    cdef double res = 0
    for j in xrange(n):
        res += y[j] * alpha[j] * z[j]
    res /= 2 * lmda
    for j in xrange(n):
        res -= alpha[j]
    return res


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
