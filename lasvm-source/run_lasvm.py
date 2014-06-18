import os

if os.name == "nt":
    ucipath = "..\\..\\dataset\\ucibenchmark"
    uspspath = "..\\..\\dataset\\usps"
    mnistpath = "..\\..\\dataset\\mnist"
elif os.name == "posix":
    ucipath = '../../dataset/benchmark_uci'
    uspspath = '../../dataset/usps/split'
    mnistpath = '../../dataset/mnist'


c = 1
pos_ind = 4
os.system("./la_svm -t 2 -c %f -b 0 %s/usps.%d usps_%d.model" % (c, uspspath, pos_ind, pos_ind))
os.system("./la_test -B 0 %s/usps.t.%d usps_%d.model usps_%d.result" % (uspspath, pos_ind, pos_ind, pos_ind))