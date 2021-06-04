from setuptools import setup, find_packages
if __name__ == '__main__':
    setup(
        name='utils',
        version='0.1.0',
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
