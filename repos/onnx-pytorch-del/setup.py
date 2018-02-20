from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
from setuptools import setup, find_packages

setup(
    name="onnx-pytorch",
    version='0.l',
    description="PyTorch helpers for working with Open Neural Network Exchange format",
    install_requires=['numpy', 'onnx'],
    setup_requires=[],
    tests_require=[],
    packages=find_packages(),
    author='ezyang',
    author_email='ezyang@fb.com',
    url='https://github.com/ezyang/onnx-pytorch',
)
