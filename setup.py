from setuptools import setup, find_packages

setup(name="plf_util",
      version="0.0.1",
      packages = [ 'plf_util' ],
      install_requires = [
        'numpy',
        'pandas',
        'matplotlib',
        'torch',
        'sklearn',
        'xlsxwriter',
        'datetime'
      ]
)
