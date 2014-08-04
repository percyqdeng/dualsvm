from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import preprocessing
import sklearn.cross_validation as cv
from sklearn.linear_model import SGDClassifier
from load_data import convert_binary, load_usps
from mylasso import *
from load_data import load_mnist, convert_binary


"""
experiments on relation between sparse pattern and lmda

"""
data = load_mnist()
pos_ind = 6
neg_ind = 5

lmda_list = [0.0005, 0.001, 0.01, 0.1, 0.3]
x, y = convert_binary(data, pos_ind, neg_ind)
n, p = x.shape
x = x.astype(float)
n_iter = 30
ss = cv.StratifiedShuffleSplit(y=y, n_iter=20, test_size=0.1, random_state=1)
nzs_scg_T = np.zeros((n_iter, len(lmda_list)))
nzs_scg_bar = np.zeros((n_iter, len(lmda_list)))
nzs_rda_T = np.zeros((n_iter, len(lmda_list)))
nzs_rda_bar = np.zeros((n_iter, len(lmda_list)))
nzs_rda2_T = np.zeros((n_iter, len(lmda_list)))
nzs_rda2_bar = np.zeros((n_iter, len(lmda_list)))
nzs_cd_T = np.zeros((n_iter, len(lmda_list)))
nzs_cd_bar = np.zeros((n_iter, len(lmda_list)))
for i, (train_idx, test_idx) in enumerate(ss):
    for j, lmda in enumerate(lmda_list):
        xtrain = x[train_idx, :]
        ytrain = y[train_idx]
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        xtrain = min_max_scaler.fit_transform(xtrain)
        ntrain = y.size
        rgr = LassoLI(lmda=lmda, b=5, c=1, T=10*ntrain, algo='scg', sig_D=100)
        rgr.fit(xtrain, ytrain)
        nzs_scg_T[i, j] = rgr.num_zs[-1]
        nzs_scg_bar[i, j] = rgr.sparsity()

        rgr2 = LassoLI(lmda=lmda, b=5, c=1, T=10*ntrain, algo='rda2', sig_D=100)
        rgr2.fit(xtrain, ytrain)
        nzs_rda2_T[i, j] = rgr2.num_zs[-1]
        nzs_rda2_bar[i, j] = rgr2.sparsity()

        rgr3 = LassoLI(lmda=lmda,  T=10*ntrain, algo='rda', sig_D=100)
        rgr3.fit(xtrain, ytrain)
        nzs_rda_T[i, j] = rgr3.num_zs[-1]
        nzs_rda_bar[i, j] = rgr3.sparsity()

        rgr4 = LassoLI(lmda=lmda, T=ntrain/20, algo='cd', sig_D=100)
        rgr4.fit(xtrain, ytrain)
        nzs_cd_T[i, j] = rgr4.num_zs[-1]
        nzs_cd_bar[i, j] = rgr4.sparsity()





