
import numpy as np
from sklearn.svm import SVC
import os

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

path = os.path.join('..', 'dataset', 'cifar', 'cifar-10-batches-py')
dfile = os.path.join(path, 'data_batch_1')

data = unpickle(dfile)
x = data['data']
y = np.asarray(data['labels'])

y[y!=0] = -1
y[y==0] = 1

clf = SVC()
clf.fit(x, y)