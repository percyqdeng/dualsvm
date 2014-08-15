import sys
import numpy as np
"""
coordinate descent on lasso
0.5/n (y-xw)^2 + lmda * ||w||_1
"""


def train(x, y, xtest=None, ytest=None, lmda=0.1, T=1000, cyc_ord='rand'):
    n, p = x.shape
    xsq = np.sum(x ** 2, axis=0)
    w = np.zeros(p)
    flag = np.ones(p)
    has_test = not(xtest is None)
    z = x.dot(w)
    if cyc_ord == 'rand':
        ind = np.random.choice(p, T, replace=True)
    elif cyc_ord == 'cyclic':
        ind = np.arange(T) % p
    else:
        print "undefined cycle order"
        exit(1)
    num_zs = []
    num_iters = []
    num_features = []
    train_obj = []
    test_obj = []
    sqnorm_w = []
    num_steps = 0
    interval = np.maximum(1, T/20)
    small_number = 1e-10
    for t, j in enumerate(ind):
        # j = (t %p)
        # print j, t
        b = -np.dot(y - (z - x[:, j] * w[j]), x[:, j]) / n
        a = xsq[j] / n
        if a < small_number:
            continue
        c = lmda
        wj_new = - np.sign(b) * np.maximum(np.fabs(b) - c, 0) / a
        flag[j] = (np.fabs(b) <= c)
        if np.isnan(wj_new):
            print 'warning, nan'
        z += x[:, j] * (wj_new - w[j])
        w[j] = wj_new
        if num_steps <= t:
            num_iters.append(t+1)
            num_features.append((t+1) * n)
            sqnorm_w.append(np.linalg.norm(w)**2)
            train_obj.append(_eval_train_obj(y, z, w, lmda))
            if has_test:
                test_obj.append(eval_lasso(xtest, ytest, w, lmda))
            num_zs.append(flag.sum())
            num_steps += interval

    if has_test:
        return w, train_obj, test_obj, num_zs, num_iters, num_features, sqnorm_w
    else:
        return w, train_obj, num_zs, num_iters, num_features, sqnorm_w


def train_effi(x, y, xtest=None, ytest=None, lmda=0.1, T=1000, cyc_ord='rand'):
    """
    more efficient coordinate with K=x^Tx
    :param x:
    :param y:
    :param xtest:
    :param ytest:
    :param lmda:
    :param T:
    :param cyc_ord:
    :return:
    """
    n, p = x.shape
    w = np.zeros(p)
    K = np.dot(x.T, x)/n
    B = np.dot(x.T, y)/n
    flag = np.ones(p)
    has_test = not(xtest is None)
    z = K.dot(w)
    if cyc_ord == 'rand':
        ind = np.random.choice(p, T, replace=True)
    elif cyc_ord == 'cyclic':
        ind = np.arange(T) % p
    else:
        print "undefined cycle order"
        exit(1)
    num_zs = []
    num_iters = []
    num_features = []
    train_obj = []
    test_obj = []
    sqnorm_w = []
    num_steps = 0
    interval = np.maximum(1, T/20)
    small_number = 1e-10
    for t, j in enumerate(ind):
        # j = (t %p)
        # print j, t
        a = K[j, j]
        b = z[j]-K[j, j]*w[j] - B[j]
        # b = -np.dot(y - (z - x[:, j] * w[j]), x[:, j])
        if a < small_number:
            continue
        c = lmda
        # if np.fabs(b) < c:
        #     wj_new = 0.0
        # else:
        #     wj_new = - np.sign(b) * (np.fabs(b)-c) /a
        wj_new = - np.sign(b) * np.maximum(np.fabs(b) - c, 0) / a
        flag[j] = (np.fabs(b) <= c)
        if np.isnan(wj_new):
            print 'warning, nan'
        z += K[:, j] * (wj_new - w[j])
        w[j] = wj_new
        if num_steps <= t:
            num_iters.append(t+1)
            num_features.append((t+1) * n)
            sqnorm_w.append(np.linalg.norm(w)**2)
            train_obj.append(_eval_train_obj2(B, z, y, w, lmda))
            if has_test:
                test_obj.append(eval_lasso(xtest, ytest, w, lmda))
            num_zs.append(flag.sum())
            num_steps += interval

    if has_test:
        return w, train_obj, test_obj, num_zs, num_iters, num_features, sqnorm_w
    else:
        return w, train_obj, num_zs, num_iters, num_features, sqnorm_w


def _eval_train_obj(y, z, w, lmda):
    return 0.5/y.size * np.sum((y - z) ** 2) + lmda * (np.linalg.norm(w, ord=1))


def _eval_train_obj2(B, z, y, w,  lmda):
    return 0.5 * w.dot(z) - B.dot(w) + lmda * (np.linalg.norm(w, ord=1)) + 0.5 * (y**2).mean()

def eval_lasso(x, y, w, lmda):
    n = y.size
    res = 0.5/n * np.linalg.norm(y-x.dot(w), ord=2)**2 + lmda*(np.linalg.norm(w, ord=1))
    return res
