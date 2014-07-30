import numpy as np


class CDLasso(object):
    """
    lasso problem:
    0.5/n||y-xw||^2 + lmda * ||w||_1
    """

    def __init__(self, lmda=1, T=1000):
        self.lmda = lmda
        self.T = T
        self.num_iters = None
        self.num_features = None
        self.num_zs = None
        self.obj = None
        self.w = None

    def train(self, x, y):
        n, p = x.shape
        lmda = n * self.lmda
        xsq = np.sum(x ** 2, axis=0)
        w = np.zeros(p)
        flag = np.zeros(p)
        z = x.dot(w)
        ind = np.random.choice(p, self.T, replace=True)
        self.num_zs = []
        self.num_iters = []
        self.num_features = []
        self.obj = []
        interval = 1
        for t, j in enumerate(ind):
            b = -np.dot(y - (z - x[:, j] * w[j]), x[:, j])
            a = xsq[j]
            c = lmda
            wj_new = - np.sign(b) * np.maximum(np.fabs(b) - c, 0) / a
            flag[j] = (np.fabs(b) <= c)

            z += x[:, j] * (wj_new - w[j])
            w[j] = wj_new
            if t == interval:
                self.num_iters.append(t+1)
                self.num_features.append((t+1) * n)
                self.obj.append(self._eval_obj(y, z, w))
                interval += 2
                self.num_zs.append(flag.sum())
        self.w = w

    def predict(self, xtest):
        return xtest.dot(self.w)

    def _eval_obj(self, y, z, w):
        return 0.5/y.size * np.sum((y - z) ** 2) + self.lmda * (np.fabs(w)).sum()
