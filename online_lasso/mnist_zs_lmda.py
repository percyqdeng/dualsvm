from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import preprocessing
import sklearn.cross_validation as cv
from sklearn import linear_model

from sklearn.linear_model import SGDClassifier
from load_data import convert_binary, load_usps
from mylasso import *
from load_data import load_mnist, convert_binary


"""
experiments on relation between sparse pattern and regularization weight lmda
"""

data = load_mnist()
pos_ind = 6
neg_ind = 5

lmda_list = [0.0005, 0.001, 0.01, 0.1, 0.3]
x, y = convert_binary(data, pos_ind, neg_ind)
n, p = x.shape
x = x.astype(float)
y = y.astype(float)
alphas, coefs, gaps = linear_model.lasso_path(x, y, n_alphas=10, return_models=False, fit_intercept=False)
lmda_list = alphas[::-1]
n_iter = 2
ss = cv.StratifiedShuffleSplit(y=y, n_iter=n_iter, test_size=0.3, random_state=1)
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

        rgr4 = linear_model.Lasso(alpha=lmda, fit_intercept=False)
        rgr4.fit(xtrain, ytrain)
        nzs_cd_T[i, j] = (rgr4.coef_==0).sum()
        nzs_cd_bar[i, j] = (rgr4.coef_==0).sum()
        # rgr4 = LassoLI(lmda=lmda, T=ntrain/20, algo='cd', sig_D=100)
        # rgr4.fit(xtrain, ytrain)
        # nzs_cd_T[i, j] = rgr4.num_zs[-1]
        # nzs_cd_bar[i, j] = rgr4.sparsity()

nzs_cd_bar = p - nzs_cd_bar
nzs_cd_T = p - nzs_cd_T

plt.figure()
plt.errorbar(np.log(lmda_list), nzs_scg_T.mean(axis=0), yerr=np.std(nzs_scg_T, axis=0), fmt='--', label='scg')
plt.errorbar(np.log(lmda_list), nzs_rda_T.mean(axis=0), yerr=np.std(nzs_rda_T, axis=0), fmt='-', label='rda')
plt.errorbar(np.log(lmda_list), nzs_rda2_T.mean(axis=0), yerr=np.std(nzs_rda2_T, axis=0), fmt='-.', label='rda2')
plt.errorbar(np.log(lmda_list), nzs_cd_T.mean(axis=0), yerr=np.std(nzs_cd_T, axis=0), fmt='-.', label='cd')
plt.ylabel(r'number of zeros in $w_T$')
plt.xlabel(r'$\lambda$')
plt.legend(loc='best')
# plt.xscale('log')
plt.savefig('../output/n_zs_T.eps')
plt.savefig('../output/n_zs_T.pdf')

plt.figure()
plt.errorbar(np.log(lmda_list), nzs_scg_bar.mean(axis=0), yerr=np.std(nzs_scg_bar, axis=0), fmt='--', label='scg')
plt.errorbar(np.log(lmda_list), nzs_rda_bar.mean(axis=0), yerr=np.std(nzs_rda_bar, axis=0), fmt='-', label='rda')
plt.errorbar(np.log(lmda_list), nzs_rda2_bar.mean(axis=0), yerr=np.std(nzs_rda2_bar, axis=0), fmt='-.', label='rda2')
plt.errorbar(np.log(lmda_list), nzs_cd_bar.mean(axis=0), yerr=np.std(nzs_cd_bar, axis=0), fmt='-.', label='cd')
plt.legend(loc='best')
# plt.xscale('log')
plt.ylabel(r'number of zeros in $\bar{w}$')
plt.xlabel(r'$\lambda$')
plt.savefig('../output/n_zs_bar.eps')
plt.savefig('../output/n_zs_bar.pdf')
