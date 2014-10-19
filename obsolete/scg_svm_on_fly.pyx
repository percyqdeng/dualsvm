# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
import time
from libcpp.vector cimport vector
# from libcpp import bool
from libc.stdlib cimport rand
# cdef extern from "math.h"
from libc.math cimport fmax, fmin, exp
from auxiliary.rand_no_repeat cimport RandNoRep
from kernel_func cimport KernelFunc

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
def scg_on_the_fly(double[:,::1] xtr, double[::1] ytr, KernelFunc kernel, double[:,::1]xte=None, double[::1]yte=None,
                      int verbose=True, double lmda=1E-5, int nsweep=1000, double rho = 1.0, int c=1, int b=5):
    """
    stochastic coordinate gradient dual average on the dual svm,
    random sample a batch of data and update on another random sampled variables
    kernel product is computed on the fly
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
    start_time = time.time()
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
            k_dot_yalpha(xtr, xtr, ytr, alpha, z, kernel)
            res = zero_one_loss(ytr, z)
            err_train.push_back(res)
            res = eval_dsvm_obj(z, ytr, alpha, lmda)
            obj.push_back(res)
            if has_test:
                k_dot_yalpha(xte, xtr, ytr, alpha, z2, kernel)
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
                k_dot_yalpha(xte, xtr, ytr, alpha, z2, kernel)
                res = zero_one_loss(yte, z2)
                err_test.push_back(res)
            # n_iters += interval
            n_iters *= scalar
    return np.asarray(alpha), np.asarray(err_train), np.asarray(err_test), np.asarray(obj_train), np.asarray(num_opers)

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
cdef double zero_one_loss(double[::1] label, double[::1] pred):
    cdef Py_ssize_t i, n
    n = label.shape[0]
    assert (n == pred.shape[0])
    cdef double err = 0
    for i in xrange(n):
        err += (label[i] * pred[i] < 0)
    return err / n


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void k_dot_yalpha(double[:,::1]xtest, double[:, ::1]x, double[::1]y, double[::1]alpha, double [::1]z, KernelFunc kernel):
    """
    the classifier output on test data
    z k.dot(y*alpha)
    """
    cdef int n = xtest.shape[0]
    cdef int m = x.shape[0]
    # cdef double [::1] c = np.zeros(n)
    cdef int i, j
    for i in xrange(n):
        z[i] = 0
        for j in xrange(m):
            z[i] += kernel.product(xtest[i], x[j]) * y[j] * alpha[j]


cdef double esti_std(np.ndarray[double, ndim=2]kk, double C, int batchsize):
    n = kk.shape[0]
    # cdef double [:] sig = np.zeros(n)
    sig = np.zeros(n)
    # alpha = np.random.uniform(0,C, n)
    rep = 100
    for i in xrange(n):
        g = kk[i, :] * C
        sig[i] = np.std(g) * n / np.sqrt(batchsize)
    return sig
