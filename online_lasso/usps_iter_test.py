from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import preprocessing
import sklearn.cross_validation as cv
from sklearn.linear_model import SGDClassifier
from load_data import convert_binary, load_usps
from mylasso import *

pos_class = 3
neg_class = 5
random_state = 33
n_iter = 1

data = load_usps()
x, y = convert_binary(data, pos_class, neg_class)
# sss = cv.StratifiedShuffleSplit(y, n_iter=n_iter, test_size=0.4, train_size=0.6, random_state=random_state)


n, p = x.shape
lmda = 0.5
xtrain, xtest, ytrain, ytest = cv.train_test_split(x, y, test_size=0.1, random_state=random_state)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
xtrain = min_max_scaler.fit_transform(xtrain)
xtest = min_max_scaler.transform(xtest)
ntrain = ytrain.size

b = 4
c = 1
eta = 0.1
sig_D = 1000
# fix number of examples
nsweep = 2
N = 4000
T1 = N
# T1 = 5
rgr = LassoLI(lmda=lmda, b=b, c=c, T=T1, algo='scg', sig_D=sig_D)
rgr.fit(xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest)
rgr.predict(xtest)

T2 = N
rgr2 = LassoLI(lmda=lmda, T=T2, algo='rda', sig_D=sig_D)
rgr2.fit(xtrain, ytrain, xtest=xtest, ytest=ytest)

T3 = N
rgr3 = LassoLI(lmda=lmda, T=T3, algo='cd')
rgr3.fit(xtrain, ytrain, xtest=xtest, ytest=ytest)
#
T4 = N
rgr4 = LassoLI(lmda=lmda, b=b, c=c, T=T4, algo='rda2', sig_D=sig_D)
rgr4.fit(xtrain, ytrain, xtest=xtest, ytest=ytest)


print "final output: \n" \
      "scg %f \n" \
      "rda %f \n" \
      "rda2 %f \n" \
      %(rgr.eval_lasso_obj(xtrain, ytrain, lmda), rgr2.eval_lasso_obj(xtrain, ytrain, lmda), rgr4.eval_lasso_obj(xtrain, ytrain, lmda))

plt.figure()
plt.subplot(221)
plt.plot(rgr.num_iters, rgr.train_obj, 'bo-', label='scg')
plt.plot(rgr2.num_iters, rgr2.train_obj, 'gd-', label='rda')
plt.plot(rgr3.num_iters, rgr3.train_obj, 'r>-', label='cd')
plt.plot(rgr4.num_iters, rgr4.train_obj, 'mD-', label='rda2')
plt.plot()
plt.xlabel('number of iters')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('optimization error')

plt.subplot(222)
plt.plot(rgr.num_iters, rgr.test_obj, 'bo-', label='scg')
plt.plot(rgr2.num_iters, rgr2.test_obj, 'gd-', label='rda')
plt.plot(rgr3.num_iters, rgr3.test_obj, 'r>-', label='cd')
plt.plot(rgr4.num_iters, rgr4.test_obj, 'mD-', label='rda2')
plt.xlabel('number of iters')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('test obj')
# plt.figure()
plt.subplot(223)
plt.plot(rgr.num_iters, rgr.num_zs, 'bo-', label='scg')
plt.plot(rgr2.num_iters, rgr2.num_zs, 'gd-', label='rda')
# plt.plot(rgr3.num_iters, rgr3.num_zs, 'r>-', label='cd')
plt.plot(rgr4.num_iters, rgr4.num_zs, 'mD-', label='rda2')
plt.xlabel('number of iters')
plt.ylabel('number of zeros')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend()
plt.show()



