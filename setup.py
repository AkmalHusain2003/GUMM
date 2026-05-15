from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

_DIRECTIVES = {
    'language_level': '3',
    'boundscheck':    False,
    'wraparound':     False,
    'cdivision':      True,
    'nonecheck':      False,
}

_INCLUDE = [np.get_include()]

extensions = [
    Extension(
        name='gumm._normalize',
        sources=['gumm/_normalize.pyx'],
        include_dirs=_INCLUDE,
    ),
    Extension(
        name='gumm._spatial',
        sources=['gumm/_spatial.pyx'],
        include_dirs=_INCLUDE,
    ),
    Extension(
        name='gumm._model',
        sources=['gumm/_model.pyx'],
        include_dirs=_INCLUDE,
    ),
]

setup(
    packages=find_packages(exclude=['tests*']),
    ext_modules=cythonize(extensions, compiler_directives=_DIRECTIVES),
    zip_safe=False,
)
