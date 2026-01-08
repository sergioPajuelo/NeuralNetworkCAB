from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="lorentzian.lorentzian",              # paquete.modulo
        sources=["lorentzian/lorentzian.pyx"],     # RUTA CORRECTA
        include_dirs=[np.get_include()],
    )
]

setup(
    name="NeuralNetworkCAB",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
