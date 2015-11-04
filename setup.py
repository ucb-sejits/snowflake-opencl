from distutils.core import setup

setup(
    name='snowflake-opencl',
    version='0.1.0',
    url='github.com/ucb-sejits/snowflake-opencl',
    license='B',
    author='Dorthy Luu',
    author_email='dluu@berkeley.edu',
    description='OpenCL compiler for snowflake',

    packages=[
        'snowflake_opencl'
    ],

    install_requires=[
        'ctree',
        'snowflake'
    ]
)
