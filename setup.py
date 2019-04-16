# -*- coding: utf-8 -*-
# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='gigan',
    version='0.0.1',
    description='Goal Inference GAN (GIGAN)',
    long_description=readme,
    author='Orlando Garcia Alvarez',
    author_email='orlando.garcia@cimat.mx',
    url='https://github.com/cimat-ris/GoalInferenceGAN',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

