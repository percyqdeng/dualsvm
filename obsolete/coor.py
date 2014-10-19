__author__ = 'qdengpercy'

import numpy as np


def stoch_coor_descent(ktr, ytr, kte, yte, lmda, nsweep, T, batchsize):
    """
    stochastic coordinate descent on the dual svm, random sample a batch of data and update on another random sampled
    variables
    """
    n = ktr.shape[0]
    cc = 1.0/n
    yktr = (ytr[:, np.newaxis] * ktr) * ytr[np.newaxis, :]
    has_kte = True
    err_tr = []
    err_te = []
    obj = []
    ker_oper = []
    nnz = []
    print"------------estimate parameters and set up variables----------------- "
    # comment:  seems very sensitive to the parameter estimation, if I use the second D_t, the algorithm diverges
    #
    lip = np.diag(yktr)/lmda
    l_max = np.max(lip)
    Q = 1
    D_t = Q * np.sqrt(1.0/(2*n))
    # D_t = Q * (n / 2) * cc
    sig_list = esti_std(yktr, lmda, cc, batchsize)
    sig = np.sqrt((sig_list ** 2).sum())
    eta = np.ones(T + 1)
    eta *= np.minimum(1.0 / (2 * l_max), D_t / sig * np.sqrt(float(n) / (1 + T)))
    theta = eta + .0
    alpha = np.zeros(n)  # the most recent solution
    a_tilde = np.zeros(n)  # the accumulated solution in the path
    delta = np.zeros(T + 2)
    uu = np.zeros(n, dtype=int)
    # index of update, u[i] = t means the most recent update of
    # ith coordinate is in the t-th round, t = 0,1,...,T
    showtimes = 5
    t = 0
    count = 0
    print "estimated sigma: "+str(sig)+" lipschitz: "+str(l_max)
    print "----------------------start the algorithm----------------------"
    for i in range(nsweep):
        # index of batch data to compute stochastic coordinate gradient
        samp = np.random.choice(n, size=(n, batchsize))
        # samp = np.random.permutation(n)
        # index of sampled coordinate to update

        perm = np.random.permutation(n)
        for j in range(n):
            # samp_ind = samp[j, :]
            samp_ind = np.take(samp, j, axis=0)
            # samp_ind = samp[j]
            var_ind = perm[j]
            # var_ind = samp_ind
            delta[t + 1] = delta[t] + theta[t]
            subk = yktr[var_ind, samp_ind]
            # stoc_coor_grad = np.dot(subk, alpha[samp_ind]) * float(n) / batchsize - 1
            stoc_coor_grad = 1/lmda*(np.dot(subk, alpha.take(samp_ind)) * float(n) / batchsize) - 1
            a_tilde[var_ind] += (delta[t + 1] - delta.take(uu.take(var_ind))) * alpha.take(var_ind)
            res = alpha.take(var_ind) - eta[t]*stoc_coor_grad
            if res < 0:
                alpha[var_ind] = 0
            elif res <= cc:
                alpha[var_ind] = res
            else:
                alpha[var_ind] = cc
            # alpha[var_ind] = np.minimum(np.maximum(0, alpha[var_ind] - eta[t]*stoc_coor_grad), cc)
            # alpha[var_ind] = _prox_mapping(g=stoc_coor_grad, x0=alpha[var_ind], r=eta[t])
            # assert(all(0 <= x <= cc for x in np.nditer(alpha[var_ind])))  #only works for size 1
            uu[var_ind] = t + 1
            t += 1
            count += batchsize
        if i % (nsweep /(showtimes)) == 0:
            print "# of sweeps " + str(i)
        #-------------compute the result after the ith sweep----------------
        if i % (5*n) == 0:
            print "# of i " + str(i)
            a_avg = a_tilde + (delta[t]-delta.take(uu)) * alpha
            a_avg /= delta[t]
            # a_avg = alpha
            # assert(all(0 <= x <= cc for x in np.nditer(a_avg)))
            yka = np.dot(yktr, a_avg)
            res = 1.0/lmda * 0.5 * np.dot(a_avg, yka) - a_avg.sum()
            obj.append(res)
            # if i > 2 and obj[-1] > obj[-2]:
            #     print "warning"
            nnzs = (a_avg != 0).sum()
            nnz.append(nnzs)
            err = np.mean(yka < 0)
            err_tr.append(err)
            ker_oper.append(count)
            if has_kte:
                pred = np.sign(np.dot(kte, ytr*a_avg))
                err = np.mean(yte != pred)
                err_te.append(err)
    # -------------compute the final result after nsweep-th sweeps---------------
    a_tilde += (delta[T + 1] - delta[uu]) * alpha
    alpha = a_tilde / delta[T + 1]
    final = lmda * (0.5 * np.dot(alpha, np.dot(yktr, alpha)) - alpha.sum())
    bound1 = (n-1)*0.5*l_max/lmda
    bound2 = l_max
    bound3 = sig * np.sqrt(2)
    return err_tr, err_te, obj, ker_oper


def esti_std(kk,  lmda, cc, batchsize):

    n = kk.shape[0]
    sig = np.zeros(n)
    alpha = np.random.uniform(0,cc, n)
    rep = 100

    for i in range(n):
        g = kk[i, :]/lmda * cc
        sig[i] = np.std(g) * n / np.sqrt(batchsize)
    return sig