from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import sys

_DIRECTIVES = {
    'language_level': '3',
    'boundscheck':    False,
    'wraparound':     False,
    'cdivision':      True,
    'nonecheck':      False,
}

_INCLUDE = [np.get_include()]

# OpenMP flags differ between GCC/Clang and MSVC.
if sys.platform == 'win32':
    _OMP_COMPILE = ['/openmp']
    _OMP_LINK    = []
else:
    _OMP_COMPILE = ['-fopenmp']
    _OMP_LINK    = ['-fopenmp']

extensions = [
    # No parallelisation — built without OpenMP to avoid unnecessary linkage.
    Extension(
        name='gumm._normalize',
        sources=['gumm/_normalize.pyx'],
        include_dirs=_INCLUDE,
    ),
    # Monte Carlo Ripley K loop parallelised with OpenMP prange.
    Extension(
        name='gumm._spatial',
        sources=['gumm/_spatial.pyx'],
        include_dirs=_INCLUDE,
        extra_compile_args=_OMP_COMPILE,
        extra_link_args=_OMP_LINK,
    ),
    # Weighted covariance (M-step) parallelised with OpenMP prange.
    Extension(
        name='gumm._model',
        sources=['gumm/_model.pyx'],
        include_dirs=_INCLUDE,
        extra_compile_args=_OMP_COMPILE,
        extra_link_args=_OMP_LINK,
    ),
]

setup(
    packages=find_packages(exclude=['tests*']),
    ext_modules=cythonize(extensions, compiler_directives=_DIRECTIVES),
    zip_safe=False,
)
