# distutils: language = c++

from libc.stdlib cimport rand
import numpy as np
cimport numpy as np


cdef class RandNoRep(object):
    """
    generate random numbers between 0 to n-1 without replacement
    """
    # cdef Py_ssize_t [::1] seeds
    # cdef Py_ssize_t n
    # cdef Py_ssize_t b_size # batch size

    def __init__(self, int n):
        self.seeds = np.zeros(n, dtype=np.intp)
        self.n = n
        cdef Py_ssize_t i
        for i in range(n):
            self.seeds[i] = i

    cdef k_choice(self, Py_ssize_t [::1] arr, Py_ssize_t k):
        """
        return k elements w/o replacement
        """
        assert (k <= self.n-1)
        assert (k <=arr.size)
        cdef Py_ssize_t p = self.n-1
        cdef Py_ssize_t i, j, tmp
        for i in range(k):
            j = rand() % self.n
            tmp = self.seeds[p]
            self.seeds[p] = self.seeds[j]
            self.seeds[j] = tmp
            arr[i] = self.seeds[p]
            p -= 1

    def k_choice_py(self, int k):
        """
        python wrapper of cython function _k_choice
        """
        arr = np.zeros(k, dtype=np.intp)
        self._k_choice(arr, k)
        return np.asarray(arr)