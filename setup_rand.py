__author__ = 'qd'


from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

extensions = ["online_lasso/rand_funcs.pyx"]
setup(
    # name='',
    include_dirs=[np.get_include()],
    ext_modules=cythonize(extensions,
                          # language="c++",
                          )
)

# if __name__ == "__main__":
#     os.

# run setup_rand.py build_ext --inplace