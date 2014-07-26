from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import preprocessing
import sklearn.cross_validation as cv
from sklearn.linear_model import SGDClassifier
from load_data import convert_binary, load_usps
from da_lasso import *
pos_class = 3
neg_class = 5
random_state = 33
n_iter = 1

data = load_usps()
x, y = convert_binary(data, pos_class, neg_class)
lmda = 0.1
b = 4
c = 4
nsweep = 10
# sss = cv.StratifiedShuffleSplit(y, n_iter=n_iter, test_size=0.4, train_size=0.6, random_state=random_state)
#
# for i, (train_index, test_index) in sss:
xtrain, xtest, ytrain, ytest = cv.train_test_split(x, y, test_size=0.8, random_state=random_state)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
xtrain = min_max_scaler.fit_transform(xtrain)
xtest = min_max_scaler.transform(xtest)

n, p = xtrain.shape
T = nsweep * n
rgr = LassoLR(lmda=lmda, b=b, c=c, T=T)
rgr.fit(xtrain, ytrain)

rgr.predict(xtest)


clf = SGDClassifier(loss="squared_loss", penalty="l1", fit_intercept=False, n_iter=nsweep, alpha=lmda)
clf.fit(xtrain, ytrain)

plt.figure()
plt.subplot(211)
plt.plot(rgr.num_iter, rgr.train_err, 'bx-')
plt.ylabel('train error')
# plt.figure()
plt.subplot(212)
plt.plot(rgr.num_iter, rgr.num_zs, 'r^-')
plt.ylabel('number of zeros')
plt.show()
