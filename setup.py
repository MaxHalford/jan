#!/usr/bin/env python3

import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='jan',
    version='0.0.1',
    author='Max Halford',
    license='MIT',
    author_email='maxhalford25@gmail.com',
    description='A simple neural network library for educational purposes.',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/MaxHalford/jan',
    packages=['jan'],
    install_requires=['numpy'],
    extras_require={'dev': ['pytest']},
    python_requires='>=3.6'
)
