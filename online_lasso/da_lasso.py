

import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import scg_lasso


class LassoLR(object):
    """
    online lasso
    """
    def __init__(self, lmda=.1, b=1, c = 5):
        self.b = 1
        self.c = 5
        self.lmda = lmda
        self.w = None

    def fit(self, xtrain, ytrain):
        n, p = xtrain.shape

        self.w = np.zeros(p)

        pass

    def predict(self, xtest):
        pass
