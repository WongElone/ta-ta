from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "traderalchemy_ta.k_zigzag",
        ["src/traderalchemy_ta/k_zigzag.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "traderalchemy_ta.k_zigzag_s",
        ["src/traderalchemy_ta/k_zigzag_s.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "traderalchemy_ta.zigzag",
        ["src/traderalchemy_ta/zigzag.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "traderalchemy_ta.zigzag_s",
        ["src/traderalchemy_ta/zigzag_s.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "traderalchemy_ta.k_pips",
        ["src/traderalchemy_ta/k_pips.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "traderalchemy_ta.pips",
        ["src/traderalchemy_ta/pips.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
]

setup(
    ext_modules=cythonize(extensions)
)
