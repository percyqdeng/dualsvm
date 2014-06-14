__author__ = 'qdengpercy'


from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='set_cmp_time',
    include_dirs=[np.get_include()],
    ext_modules=cythonize("src/cmp_time.pyx",
                          language="c++",
                          )
)

# if __name__ == "__main__":
#     os.

# run set_cmp_time.py build_ext --inplace
