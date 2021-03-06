#!/usr/bin/env python

# from distutils.core import setup

import os
# To use a consistent encoding
from codecs import open

import setuptools

# Get the long description from the README file
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'README.md')) as file:
    long_description = file.read()

setuptools.setup(

    name='trapy',

    version='0.6.3',

    description='Structural Fire Engineering - Probabilistic Reliability Assessment',

    author='Yan Fu',

    author_email='fuyans@gmail.com',

    url='https://github.com/fsepy/sfeprapy',

    download_url="https://github.com/fsepy/sfeprapy/archive/master.zip",

    keywords=["fire safety", "structural fire engineering"],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],

    long_description='Structural Fire Engineering - Probabilistic Reliability Assessment',

    packages=[
        'trapy',
        'trapy.func',
        ],

    install_requires=[
        'matplotlib>=2.2.2',
        'numpy>=1.15.0',
        'pandas>=0.23.3',
        'scipy>=1.1.0',
        'seaborn>=0.9.0',
        'tqdm',
    ],

    include_package_data=True,
)
