from setuptools import setup
from Cython.Build import cythonize

setup(
    name='hpcfinal',
    description='Librería para métricas de clasificación y regresión',
    author='María Paula Parra',
    packages=['hpcfinal'],
    ext_modules=cythonize(['src/hpcfinal/classification/metrics.pyx',
                           'src/hpcfinal/regression/metrics.pyx']),
    zip_safe=False,
)
