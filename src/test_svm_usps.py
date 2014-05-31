__author__ = 'qdengpercy'


from dualksvm import *
from stocksvm import *
from sklearn import svm
import os
import time
import numpy as np
import load_data.load_usps as load_usps


def convert_binary(data, k):
    """
    convert 0-9 digits to binary dataset
    """
    assert 0 <= k <= 9
    x_pos = data[str(k)]

    n_neg = 0
    x_neg = None
    for i in range(10):
        n_neg += data[str(i)].shape[0]
        if i != k:
            if x_neg is None:
                x_neg = data[str(i)]
            else:
                x_neg = np.vstack((x_neg, data[str(i)]))
    x = np.vstack((x_pos, x_neg))
    y = np.ones(x.shape[0], dtype=np.int)
    y[x_pos.shape[0]:-1] = -1
    return x, y

if __name__ == '__main__':
    k = 4
    data = load_usps()





