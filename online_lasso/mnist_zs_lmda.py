from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import preprocessing
import sklearn.cross_validation as cv
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
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
sig_D = 150
# lmda_list = [0.0005, 0.001, 0.01, 0.1, 0.3]
x, y = convert_binary(data, pos_ind, neg_ind)
n, p = x.shape
x = x.astype(float)
y = y.astype(float)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
x = min_max_scaler.fit_transform(x)
# xtest = min_max_scaler.transform(x)
# ntrain = ytrain.size

random_state =  np.random.randint(low=1,high=10000)
alphas, coefs, gaps = linear_model.lasso_path(x, y, n_alphas=6, return_models=False, fit_intercept=False)
lmda_list = alphas[::-1]
lmda_list = lmda_list[1::]
rho_list = np.array([80, 80, 80, 100, 140, 180])
n_iter = 4
ss = cv.StratifiedShuffleSplit(y=y, n_iter=n_iter, test_size=0.3, random_state=random_state)
nzs_scg1_T = np.zeros((n_iter, len(lmda_list)))
nzs_scg1_bar = np.zeros((n_iter, len(lmda_list)))
nzs_scg2_T = np.zeros((n_iter, len(lmda_list)))
nzs_scg2_bar = np.zeros((n_iter, len(lmda_list)))
nzs_scg3_T = np.zeros((n_iter, len(lmda_list)))
nzs_scg3_bar = np.zeros((n_iter, len(lmda_list)))
nzs_rda_T = np.zeros((n_iter, len(lmda_list)))
nzs_rda_bar = np.zeros((n_iter, len(lmda_list)))
nzs_rda0_T = np.zeros((n_iter, len(lmda_list)))
nzs_rda0_bar = np.zeros((n_iter, len(lmda_list)))
nzs_rda2_T = np.zeros((n_iter, len(lmda_list)))
nzs_rda2_bar = np.zeros((n_iter, len(lmda_list)))
nzs_cd_T = np.zeros((n_iter, len(lmda_list)))
nzs_cd_bar = np.zeros((n_iter, len(lmda_list)))
nzs_sgd = np.zeros((n_iter, len(lmda_list)))
obj_scg1_T = np.zeros((n_iter, len(lmda_list)))
obj_scg1_bar = np.zeros((n_iter, len(lmda_list)))
obj_rda2_T = np.zeros((n_iter, len(lmda_list)))
obj_rda2_bar = np.zeros((n_iter, len(lmda_list)))
obj_rda_T = np.zeros((n_iter, len(lmda_list)))
obj_rda_bar = np.zeros((n_iter, len(lmda_list)))
obj_cd_T = np.zeros((n_iter, len(lmda_list)))
obj_cd_bar = np.zeros((n_iter, len(lmda_list)))
nsweep = 1
b = 5
blist = np.array([4, 24, 44])
c = 1
total_features = n *0.7 * (b+c)
for i, (train_idx, test_idx) in enumerate(ss):
    print "#iter: %d" % i
    xtrain = x[train_idx, :]
    ytrain = y[train_idx]
    ntrain = ytrain.size
    for j, (sig_D,lmda) in enumerate(zip(rho_list,lmda_list)):
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # xtrain = min_max_scaler.fit_transform(xtrain)
        T1 =total_features/(blist[0]+c)+1
        scg = LassoLI(lmda=lmda, b=blist[0], c=c, T=total_features/(b+c), algo='scg', sig_D=sig_D)
        scg.fit(xtrain, ytrain)
        nzs_scg1_T[i, j] = scg.num_zs[-1]
        nzs_scg1_bar[i, j] = scg.sparsity()
        obj_scg1_T[i, j] = scg.train_obj[-1]
        obj_scg1_bar[i, j] = scg.eval_lasso_obj(xtrain, ytrain, lmda)
        T2 = total_features/(blist[0]+c)+1

        # scg = LassoLI(lmda=lmda, b=blist[1], c=c, T=ntrain, algo='scg', sig_D=sig_D)
        # scg.fit(xtrain, ytrain)
        # nzs_scg2_T[i, j] = scg.num_zs[-1]
        # nzs_scg2_bar[i, j] = scg.sparsity()


        # T3 = total_features/(blist[0]+c)+1,
        # scg = LassoLI(lmda=lmda, b=blist[2], c=c, T=ntrain, algo='scg', sig_D=sig_D)
        # scg.fit(xtrain, ytrain)
        # nzs_scg3_T[i, j] = scg.num_zs[-1]
        # nzs_scg3_bar[i, j] = scg.sparsity()

        rda2 = LassoLI(lmda=lmda, b=blist[0], c=c, T=total_features/(b+c), algo='rda2', sig_D=sig_D)
        rda2.fit(xtrain, ytrain)
        nzs_rda2_T[i, j] = rda2.num_zs[-1]
        nzs_rda2_bar[i, j] = rda2.sparsity()
        obj_rda2_T[i, j] = rda2.train_obj[-1]
        obj_rda2_bar[i, j] = rda2.eval_lasso_obj(xtrain, ytrain, lmda)

        rda = LassoLI(lmda=lmda,  T=total_features/p, algo='rda', sig_D=sig_D)
        rda.fit(xtrain, ytrain)
        nzs_rda_T[i, j] = rda.num_zs[-1]
        nzs_rda_bar[i, j] = rda.sparsity()
        # obj_rda_T[i, j] = rda.train_obj[-1]
        # obj_rda_bar[i, j] = rda.eval_lasso_obj(xtrain, ytrain, lmda)

        rda0 = LassoLI(lmda=lmda,  T=2*ntrain, algo='rda', sig_D=sig_D)
        rda0.fit(xtrain, ytrain)
        nzs_rda0_T[i, j] = rda0.num_zs[-1]
        nzs_rda0_bar[i, j] = rda0.sparsity()

        # rgr4 = linear_model.Lasso(alpha=lmda, fit_intercept=False, normalize=False)
        # rgr4.fit(xtrain, ytrain)
        # nzs_cd_T[i, j] = (rgr4.coef_==0).sum()
        # nzs_cd_bar[i, j] = (rgr4.coef_==0).sum()
        rgr4 = LassoLI(lmda=lmda, T=ntrain/20, algo='cd', cd_ord='rand')
        rgr4.fit(xtrain, ytrain)
        nzs_cd_T[i, j] = rgr4.num_zs[-1]
        nzs_cd_bar[i, j] = rgr4.sparsity()

        total = b*c*nsweep*ntrain/p
        rgr5 = SGDRegressor(loss="squared_loss", penalty='l1', alpha=2*lmda, fit_intercept=False, n_iter=1)
        samp_indx = np.random.choice(ntrain, total, replace=False)
        rgr5.fit(xtrain[samp_indx, :], ytrain[samp_indx])
        nzs_sgd[i, j] = (rgr5.coef_==0).sum()

# nzs_cd_bar = p - nzs_cd_bar
# nzs_cd_T = p - nzs_cd_T

plt.figure()
plt.errorbar(np.log(lmda_list), nzs_scg1_T.mean(axis=0), yerr=np.std(nzs_scg1_T, axis=0), fmt='D--', label='scg')
# plt.errorbar(np.log(lmda_list), nzs_scg2_T.mean(axis=0), yerr=np.std(nzs_scg2_T, axis=0), fmt='D--', label='scg k=%d' % (blist[1]+1))
# plt.errorbar(np.log(lmda_list), nzs_scg3_T.mean(axis=0), yerr=np.std(nzs_scg3_T, axis=0), fmt='D--', label='scg k=%d' % (blist[2]+1))
plt.errorbar(np.log(lmda_list), nzs_rda0_T.mean(axis=0), yerr=np.std(nzs_rda0_T, axis=0), fmt='o-.', label='rda0')
plt.errorbar(np.log(lmda_list), nzs_rda_T.mean(axis=0), yerr=np.std(nzs_rda_T, axis=0), fmt='x--', label='rda')
plt.errorbar(np.log(lmda_list), nzs_rda2_T.mean(axis=0), yerr=np.std(nzs_rda2_T, axis=0), fmt='<-.', label='rda2')
plt.errorbar(np.log(lmda_list), nzs_cd_T.mean(axis=0), yerr=np.std(nzs_cd_T, axis=0), fmt='s-.', label='rcd')
plt.errorbar(np.log(lmda_list), nzs_sgd.mean(axis=0), yerr=np.std(nzs_sgd, axis=0), fmt=',-.', label='sgd')
plt.ylabel(r'number of zeros in $w_T$')
plt.xlabel(r'$\lambda$')
plt.legend(loc='best')
# plt.xscale('log')
plt.savefig('../output/n_zs_T_rho_%d.eps' % sig_D)
plt.savefig('../output/n_zs_T_rho_%d.pdf' % sig_D)

plt.figure()
plt.errorbar(np.log(lmda_list), nzs_scg1_bar.mean(axis=0), yerr=np.std(nzs_scg1_bar, axis=0), fmt='D--', label='scg k=%d' % (blist[0]+1))
# plt.errorbar(np.log(lmda_list), nzs_scg2_bar.mean(axis=0), yerr=np.std(nzs_scg2_bar, axis=0), fmt='D--', label='scg k=%d' % (blist[1]+1))
# plt.errorbar(np.log(lmda_list), nzs_scg3_bar.mean(axis=0), yerr=np.std(nzs_scg3_bar, axis=0), fmt='D--', label='scg k=%d' % (blist[2]+1))
plt.errorbar(np.log(lmda_list), nzs_rda0_bar.mean(axis=0), yerr=np.std(nzs_rda0_bar, axis=0), fmt='o--', label='rda0')
plt.errorbar(np.log(lmda_list), nzs_rda_bar.mean(axis=0), yerr=np.std(nzs_rda_bar, axis=0), fmt='x--', label='rda')
plt.errorbar(np.log(lmda_list), nzs_rda2_bar.mean(axis=0), yerr=np.std(nzs_rda2_bar, axis=0), fmt='<-.', label='rda2')
plt.errorbar(np.log(lmda_list), nzs_cd_bar.mean(axis=0), yerr=np.std(nzs_cd_bar, axis=0), fmt='s-.', label='rcd')
plt.legend(loc='best')
# plt.xscale('log')
plt.ylabel(r'number of zeros in $\bar{w}$')
plt.xlabel(r'$\lambda$')
plt.savefig('../output/n_zs_bar_rho_%d.eps' % sig_D)
plt.savefig('../output/n_zs_bar_rho_%d.pdf' % sig_D)

plt.figure()
plt.errorbar(np.log(lmda_list), obj_scg1_bar.mean(axis=0), yerr=np.std(obj_scg1_bar, axis=0), fmt='D--', label='scg k=%d' % (blist[0]+1))
# plt.errorbar(np.log(lmda_list), nzs_scg2_bar.mean(axis=0), yerr=np.std(nzs_scg2_bar, axis=0), fmt='D--', label='scg k=%d' % (blist[1]+1))
# plt.errorbar(np.log(lmda_list), nzs_scg3_bar.mean(axis=0), yerr=np.std(nzs_scg3_bar, axis=0), fmt='D--', label='scg k=%d' % (blist[2]+1))
plt.errorbar(np.log(lmda_list), obj_rda_bar.mean(axis=0), yerr=np.std(obj_rda_bar, axis=0), fmt='x--', label='rda')
plt.errorbar(np.log(lmda_list), obj_rda2_bar.mean(axis=0), yerr=np.std(obj_rda2_bar, axis=0), fmt='<-.', label='rda2')
plt.errorbar(np.log(lmda_list), obj_cd_bar.mean(axis=0), yerr=np.std(obj_cd_bar, axis=0), fmt='s-.', label='rcd')
# plt.legend(loc='best')
# plt.xscale('log')



plt.figure()
plt.errorbar(np.log(lmda_list), obj_scg1_T.mean(axis=0), yerr=np.std(obj_scg1_T, axis=0), fmt='D--', label='scg k=%d' % (blist[0]+1))
# plt.errorbar(np.log(lmda_list), nzs_scg2_T.mean(axis=0), yerr=np.std(nzs_scg2_T, axis=0), fmt='D--', label='scg k=%d' % (blist[1]+1))
# plt.errorbar(np.log(lmda_list), nzs_scg3_T.mean(axis=0), yerr=np.std(nzs_scg3_T, axis=0), fmt='D--', label='scg k=%d' % (blist[2]+1))
plt.errorbar(np.log(lmda_list), obj_rda_T.mean(axis=0), yerr=np.std(obj_rda_T, axis=0), fmt='x--', label='rda')
plt.errorbar(np.log(lmda_list), obj_rda2_T.mean(axis=0), yerr=np.std(obj_rda2_T, axis=0), fmt='<-.', label='rda2')
plt.errorbar(np.log(lmda_list), obj_cd_T.mean(axis=0), yerr=np.std(obj_cd_T, axis=0), fmt='s-.', label='rcd')
