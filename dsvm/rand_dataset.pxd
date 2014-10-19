# distutils: language = c++
# Author: qdeng

cimport cython
from libc.limits cimport INT_MAX
cimport numpy as np
import numpy as np
from kernel_func cimport KernelFunc
# np.import_array()
from libcpp cimport bool

cdef class RandomDataset(object):
    """Base class for datasets with sequential data access. """

    cdef KernelFunc kernel_function
    cdef double [:, ::1] xtrain,
    cdef double [:, ::1]xtest
    cdef double [:, ::1] ktrain
    cdef double [:, ::1] ktest
    cdef double [::1] ytrain
    cdef double [::1] ytest
    cdef Py_ssize_t n_train, n_test,
    cdef bool is_precomputed, has_test_data

    cdef double kernel_entry(self, int i, int j)
    cdef void ktrain_dot_yalpha(self, double[::1]alpha, double [::1]z)
    cdef void ktest_dot_yalpha(self, double[::1]alpha, double [::1]z)


