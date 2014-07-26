

cdef class RandNoRep(object):
    cdef Py_ssize_t [::1] seeds
    cdef  Py_ssize_t n
    cdef Py_ssize_t b_size # batch size
    cdef k_choice(self, Py_ssize_t [::1] arr, Py_ssize_t k)
    # cdef k_choice_py(self, int k)