__author__ = 'qdengpercy'


from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

extensions = ["src/coor_cy.pyx"]
setup(
    # name='stoc_coor_cython',
    include_dirs=[np.get_include()],
    ext_modules=cythonize(extensions,
                          language="c++",
                          )
)

# if __name__ == "__main__":
#     os.

# run setup.py build_ext --inplace