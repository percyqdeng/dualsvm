from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import preprocessing
import sklearn.cross_validation as cv
from sklearn.linear_model import SGDClassifier
from load_data import convert_binary, load_usps
from mylasso import *
from load_data import load_mnist, convert_binary

"""
experiments to relation between test number of features and optimization error
"""

data = load_mnist()
pos_ind = 3
neg_ind = 5

x, y = convert_binary(data, pos_ind, neg_ind)

# xsum = x.sum(axis=0)
# ind = np.where(xsum>0)  # return object is tuple
# x = x[:, ind[0]]
x = x.astype(float)
n, p = x.shape
random_state = 2
lmda = 0.01
xtrain, xtest, ytrain, ytest = cv.train_test_split(x, y, test_size=0.1, random_state=random_state)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
xtrain = min_max_scaler.fit_transform(xtrain)
xtest = min_max_scaler.transform(xtest)
ntrain = ytrain.size

b = 4
c = 1
eta = 0.1
sig_D = 100
# fix number of examples
nsweep = 2
T1 = nsweep * ntrain
rgr = LassoLI(lmda=lmda, b=b, c=c, T=T1, algo='scg', sig_D=sig_D)
rgr.fit(xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest)
rgr.predict(xtest)

T2 = nsweep * ntrain * b * c / p
rgr2 = LassoLI(lmda=lmda, T=T2, algo='rda', sig_D=sig_D)
rgr2.fit(xtrain, ytrain, xtest=xtest, ytest=ytest)

T3 = nsweep * b * c
rgr3 = LassoLI(lmda=lmda, T=T3, algo='cd')
rgr3.fit(xtrain, ytrain, xtest=xtest, ytest=ytest)

T4 = nsweep * ntrain
rgr4 = LassoLI(lmda=lmda, b=b, c=c, T=T4, algo='rda2', sig_D=sig_D)
rgr4.fit(xtrain, ytrain, xtest=xtest, ytest=ytest)


print "error of w_bar: \n" \
      "scg %f \n" \
      "rda %f \n" \
      "rda2 %f \n" \
      %(rgr.eval_lasso_obj(xtrain, ytrain, lmda), rgr2.eval_lasso_obj(xtrain, ytrain, lmda), rgr4.eval_lasso_obj(xtrain, ytrain, lmda))

print "nzs of w_bar: \n" \
      "scg %f \n" \
      "rda %f \n" \
      "rda2 %f \n" \
    %(rgr.sparsity(), rgr2.sparsity(), rgr4.sparsity())

row = 1
col = 2
plt.figure()
plt.subplot(row, col, 1)
plt.plot(rgr.num_features, rgr.train_obj, 'bo-', label='scg')
plt.plot(rgr2.num_features, rgr2.train_obj, 'gd-', label='rda')
plt.plot(rgr3.num_features, rgr3.train_obj, 'r>-', label='cd')
plt.plot(rgr4.num_features, rgr4.train_obj, 'mD-', label='rda2')
plt.plot()
plt.xlabel('number of features')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('optimization error')

plt.subplot(row, col, 2)
plt.plot(rgr.num_features, rgr.num_zs, 'bo-', label='scg')
plt.plot(rgr2.num_features, rgr2.num_zs, 'gd-', label='rda')
plt.plot(rgr3.num_features, rgr3.num_zs, 'r>-', label='cd')
plt.plot(rgr4.num_features, rgr4.num_zs, 'mD-', label='rda2')
plt.xlabel('number of features')
plt.ylabel('number of zeros')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# plt.legend(bbox_to_anchor=(-1.3, 1.08, 2.2, .1), loc=2, ncol=2, mode="expand", borderaxespad=0.)
plt.legend()
plt.tight_layout()
plt.savefig('../output/mnist_feature_lasso.eps')
plt.savefig('../output/mnist_feature_lasso.pdf')






