__author__ = 'qdengpercy'


from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='stoc_coor_cython',
    ext_modules = cythonize("coor_cy.pyx")
)


# run setup.py build_ext --inplace