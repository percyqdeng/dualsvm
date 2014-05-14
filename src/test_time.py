__author__ = 'qdengpercy'


import numpy as np
import time

def cmp_perm_uniform():
    # compute time of permutation and uniform without replacement
    start = time.time()
    n = 100000
    sample = np.random.permutation(n)
    cost1 = time.time() - start

    start = time.time()
    sample = np.random.choice(n, n, replace=False)
    cost2 = time.time() - start

    print "time cost"
    print "rand perm "+str(cost1)
    print "rand int w/o replace "+str(cost2)


if __name__ =="__main__":
    cmp_perm_uniform()
