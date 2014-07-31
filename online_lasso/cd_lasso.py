import numpy as np




def train(x, y, xtest=None, ytest=None, lmda=0.1, T=1000):
    n, p = x.shape
    xsq = np.sum(x ** 2, axis=0)
    w = np.zeros(p)
    flag = np.zeros(p)
    has_test = not(xtest is None)
    z = x.dot(w)
    ind = np.random.choice(p, T, replace=True)
    num_zs = []
    num_iters = []
    num_features = []
    num_steps = 0
    train_obj = []
    test_obj = []
    interval = 200
    for t, j in enumerate(ind):
        b = -np.dot(y - (z - x[:, j] * w[j]), x[:, j])
        a = xsq[j]
        c = lmda
        wj_new = - np.sign(b) * np.maximum(np.fabs(b) - c, 0) / a
        flag[j] = (np.fabs(b) <= c)

        z += x[:, j] * (wj_new - w[j])
        w[j] = wj_new
        if t == num_steps:
            num_iters.append(t+1)
            num_features.append((t+1) * n)
            train_obj.append(_eval_train_obj(y, z, w, lmda))
            if has_test:
                test_obj.append(eval_lasso(xtest, ytest, w, lmda))
            num_zs.append(flag.sum())
            num_steps += interval


def _eval_train_obj(y, z, w, lmda):
    return 0.5/y.size * np.sum((y - z) ** 2) + lmda * (np.fabs(w)).sum()


def eval_lasso(x, y, w, lmda):
    n = y.size
    res = 0.5/n * np.linalg.norm(y-x.dot(w), ord=2)**2 + lmda*(np.linalg.norm(w, ord=1))
    return res
