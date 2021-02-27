#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################
#  _   __      _   _               _             #
# | | / /     | | | |             (_)            #
# | |/ /  __ _| |_| |__   ___ _ __ _ _ __   ___  #
# |    \ / _` | __| '_ \ / _ \ '__| | '_ \ / _ \ #
# | |\  \ (_| | |_| | | |  __/ |  | | | | |  __/ #
# \_| \_/\__,_|\__|_| |_|\___|_|  |_|_| |_|\___| #
#                                                #
# General Video Game AI                          #
# Copyright (C) 2020-2021 d33are                 #
##################################################

from setuptools import setup

setup(name='kat_framework',
      version='1.0.0',
      description='Katherine General Video Game AI',
      url='http://github.com/d33are/kat-framework',
      author='d33are',
      author_email='d33are@gmail.com',
      license='MIT',
      packages=['kat_typing', 'kat_api', 'kat_framework', 'kat_tensorflow'],
      include_package_data=True,
      install_requires=[
            'numpy',
            'scikit-image',
            'uritools',
            'overrides',
            'portpicker',
            'himl',
      ],
      zip_safe=False)
