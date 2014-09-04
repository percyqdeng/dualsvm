from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import sklearn.cross_validation as cv
from sklearn.metrics import zero_one_loss
import numpy as np





def tune_adaboost(x, y, Tlist=[50, 80, 100, 130]):
    n, p = x.shape
    kf = cv.KFold(n, n_folds=3)
    err = np.zeros(len(Tlist))
    for train_ind, valid_ind in kf:
        xtrain = x[train_ind, :]
        ytrain = y[train_ind]
        ntrain = ytrain.size
        xvalid = x[valid_ind, :]
        yvalid = y[valid_ind]
        for i, t in enumerate(Tlist):
            clf = AdaBoostClassifier(n_estimators=t)
            clf.fit(xtrain, ytrain)
            pred = clf.predict(xvalid)
            err[i] += zero_one_loss(pred, yvalid)

    return Tlist[np.argmin(err)]


def tune_libsvm(x, y, gmlist=[1], Clist=[1]):
    # cross validation to tweak the parameter
    n, p = x.shape
    err = np.zeros((len(gmlist), len(Clist)))
    kf = cv.KFold(n, n_folds=3)
    for train_ind, valid_ind in kf:
        xtrain = x[train_ind, :]
        ytrain = y[train_ind]
        xvalid = x[valid_ind, :]
        yvalid = y[valid_ind]
        for i, gm in enumerate(gmlist):
            for j, C in enumerate(Clist):
                clf = svm.SVC(C=C, kernel='rbf', gamma=gm, )
                clf.fit(xtrain, ytrain)
                pred = clf.predict(xvalid)
                err[i, j] += zero_one_loss(pred, yvalid)
    row, col = np.unravel_index(err.argmin(), err.shape)
    return gmlist[row], Clist[col]