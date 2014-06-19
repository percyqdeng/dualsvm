import os


n = 7329
c = 1.36E-2 * 7329
pos_ind = 8
gm = 2.0/n

if os.name == "nt":
    ucipath = "..\\..\\dataset\\ucibenchmark"
    uspspath = "..\\..\\dataset\\usps\\split"
    mnistpath = "..\\..\\dataset\\mnist"
    os.system("la_svm -t 2 -g %f -c %f -b 0 %s\\usps.%d model\\usps_%d.model" % (gm, c, uspspath, pos_ind, pos_ind))
    os.system("la_test -B 0 %s\\usps.t.%d model\\usps_%d.model usps_%d.result" % (uspspath, pos_ind, pos_ind, pos_ind))
elif os.name == "posix":
    ucipath = '../../dataset/benchmark_uci'
    uspspath = '../../dataset/usps/split'
    mnistpath = '../../dataset/mnist'

    os.system("./la_svm -t 2 -g %f -c %f -b 0 %s/usps.%d model/usps_%d.model" % (gm, c, uspspath, pos_ind, pos_ind))
    os.system("./la_test -B 0 %s/usps.t.%d model/usps_%d.model usps_%d.result" % (uspspath, pos_ind, pos_ind, pos_ind))