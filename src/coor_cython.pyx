__author__ = 'qdengpercy'

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp import bool
from libc.stdlib cimport rand
cdef stoc_coor(a, ktr, lmda):
    """
    min 0.5/lmda * a'*kk*a - a'*1
    sub to 0<= a <=1/n
    """
    n = ktr.shape[0]


    return


cpdef cython_stoc_coor(np.ndarray[double, ndim=2]ktr, np.ndarray[double, ndim=1]ytr,
                     np.ndarray[double, ndim=2]kte, np.ndarray[double, ndim=1]yte,
                     double lmda, int nsweep, int T, int batchsize):
    """
    stochastic coordinate descent  on the dual svm, random sample a batch of data and update on another random sampled
    variables
    min 0.5/kmda* a'*A*a - a'*1
    sub to 0<= a <=cc

    alpha :
    a_tilde :
    """
    cdef n = ktr.shape[0]
    cdef np.ndarray[double, ndim=2] yktr = (ytr[:, np.newaxis] * ktr) * ytr[np.newaxis, :]
    cdef vector[int] nnzs
    cdef vector[double]err_tr
    cdef vector[int] ker_oper
    cdef vector[double] obj
    cdef bool has_kte = True
    cdef vector[double] err_te
    print"------------estimate parameters and set up variables----------------- "
    # comment:  seems very sensitive to the parameter estimation, if I use the second D_t, the algorithm diverges
    #
    cdef double cc = 1.0 / n
    cdef np.ndarray[double] lip = np.diag(yktr)/lmda
    cdef double l_max = np.max(lip)
    cdef double Q = 1
    cdef double D_t = Q * np.sqrt(1.0/(2*n))
    sig_list = esti_std(yktr, lmda, cc, batchsize)
    sig = np.sqrt((sig_list ** 2).sum())
    eta = np.ones(T + 1)
    eta *= np.minimum(1.0 / (2 * l_max), D_t / sig * np.sqrt(float(n) / (1 + T)))
    theta = eta + .0
    alpha = np.zeros(n)
    cdef np.ndarray[double, ndim=1] a_tilde = np.zeros(n)
    delta = np.zeros(T + 2)
    uu = np.zeros(n, dtype=int)
    # index of update, u[i] = t means the most recent update of
    # ith coordinate is in the t-th round, t = 0,1,...,T
    cdef int showtimes = 5
    cdef Py_ssize_t t = 0
    cdef Py_ssize_t i, j, k
    cdef int count = 0
    cdef np.ndarray[int] samp_ind
    # print "estimated sigma: "+str(sig)+" lipschitz: "+str(l_max)
    print "----------------------start the algorithm----------------------"
    cdef Py_ssize_t var_ind
    cdef double stoc_coor_grad
    cdef np.ndarray[int,ndim=1] samp_ind = np.zeros(batchsize)
    for i in range(nsweep):
        # index of batch data to compute stochastic coordinate gradient
        samp = np.random.choice(n, size=(n, batchsize))
        # samp = np.random.permutation(n)
        # index of sampled coordinate to update

        perm = np.random.permutation(n)
        cdef int j
        for j in range(n):
            # samp_ind = samp[j, :]
            for k in range(batchsize):
                samp_ind[k] = int(rand() % n)
            # samp_ind = samp[j]
            var_ind = perm[j]
            # var_ind = samp_ind
            delta[t + 1] = delta[t] + theta[t]
            stoc_coor_grad = 0
            for k in range(batchsize):

            subk = yktr[var_ind, samp_ind]

            stoc_coor_grad = 1/lmda*(np.dot(subk, alpha[samp_ind]) * n / batchsize) - 1
            a_tilde[var_ind] += (delta[t + 1] - delta[uu[var_ind]]) * alpha[var_ind]
            res = alpha[var_ind] - eta[t]*stoc_coor_grad
            if res < 0:
                alpha[var_ind] = 0
            elif res <= cc:
                alpha[var_ind] = res
            else:
                alpha[var_ind] = cc
            # alpha[var_ind] = np.minimum(np.maximum(0, alpha[var_ind] - eta[t]*stoc_coor_grad), _cc)
            # alpha[var_ind] = _prox_mapping(g=stoc_coor_grad, x0=alpha[var_ind], r=eta[t])
            # assert(all(0 <= x <= _cc for x in np.nditer(alpha[var_ind])))  #only works for size 1
            uu[var_ind] = t + 1
            t += 1
            count += batchsize
        if i % (nsweep / 100 * showtimes) == 0:
            print "# of sweeps " + str(i)
        #-------------compute the result after the ith sweep----------------
        a_avg = a_tilde + (delta[t]-delta[uu]) * alpha
        a_avg /= delta[t]
        # a_avg = alpha
        # assert(all(0 <= x <= _cc for x in np.nditer(a_avg)))
        yka = np.dot(yktr, a_avg)
        res = 1.0/lmda * 0.5 * np.dot(a_avg, yka) - a_avg.sum()
        # obj.append(res)
        obj.push_back(res)
        # nnzs.append((a_avg != 0).sum())
        nnzs.push_back((a_avg != 0).sum())
        err = np.mean(yka < 0)
        # err_tr.append(err)
        err_tr.push_back(err)
        # ker_oper.append(count)
        ker_oper.push_back(count)
        if has_kte:
            pred = np.sign(np.dot(kte, ytr*a_avg))
            err = np.mean(yte != pred)
            err_te.append(err)
    # -------------compute the final result after nsweep-th sweeps---------------
    a_tilde += (delta[T + 1] - delta[uu]) * alpha
    alpha = a_tilde / delta[T + 1]
    return alpha, err_tr, err_te, obj, ker_oper


cdef esti_std(np.ndarray[double, ndim=2]kk, double lmda, double cc, Py_ssize_t batchsize):

    cdef Py_ssize_t n = kk.shape[0]
    cdef np.ndarray[double] sig = np.zeros(n)
    alpha = np.random.uniform(0,cc, n)
    rep = 100
    cdef Py_ssize_t i
    for i in range(n):
        g = kk[i, :]/lmda * cc
        sig[i] = np.std(g) * n / np.sqrt(batchsize)
    return sig