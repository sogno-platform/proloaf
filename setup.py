from setuptools import setup, find_packages

setup(name="fc_util",
      version="0.0.1",
      packages = [ 'fc_util' ],
      install_requires = [
        'numpy'
        'pandas'
        'matplotlib'
        'torch'
        'sklearn'
        'csv'
        'xlsxwriter'
        'datetime'
        'math'   
      ]
)

