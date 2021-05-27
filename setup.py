from setuptools import setup, find_packages
if __name__ == '__main__':
    setup(
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
