from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name             = 'mocu',
      version          = '0.1',
      author           = 'Anthony M. DeGennaro',
      author_email     = 'adegennaro@bnl.gov',
      description      = 'Python tools for Mean Objective Cost of Uncertainty (MOCU)',
      long_description = readme(),
      classifiers      = [
        'Topic :: Optimal Experimental Design :: MOCU',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: ISC License',
        'Programming Language :: Python :: 3.6',
      ],
      keywords         = 'mocu experimental design robust optimization uncertainty quantification',
      url              = 'http://github.com/adegenna/mocu',
      license          = 'ISC',
      packages         = ['mocu','mocu.src','mocu.utils','mocu.tests','mocu.verification_tests','mocu.scripts'],
      package_dir      = {'mocu'         : 'mocu' , \
                          'mocu.src'     : 'mocu/src' ,\
                          'mocu.utils'   : 'mocu/utils' , \
                          'mocu.tests'   : 'mocu/tests' , \
                          'mocu.verification_tests'   : 'mocu/verification_tests' , \
                          'mocu.scripts' : 'mocu/scripts'},
      test_suite       = 'mocu.tests',
      entry_points     = { 'console_scripts': ['Package = mocu.scripts.example:main' ], \
                           'console_scripts': ['Package = mocu.scripts.visualizetoysystem:main' ] , \
                           'console_scripts': ['Package = mocu.verification_tests.test_dehghannasiri:main' ] },
      install_requires = [ 'numpy', 'scipy', 'matplotlib', 'sdeint', 'configparser' , 'torch' ],
      python_requires  = '>=3',
      zip_safe         = False
)
