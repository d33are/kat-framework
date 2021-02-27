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

from typing import TypeVar

# numeric types
TrainLoss = TypeVar('TrainLoss')
# array like objects with shape and dtype
Tensor = TypeVar('Tensor')
# neural network layer activation function
Activation = TypeVar('Activation')
# agent action from the action space
Action = TypeVar('Action')
# network policy
Policy = TypeVar('Policy')
# metric data rename
MetricData = TypeVar('MetricData')
# any kind of iterable dataset
IterableDataset = TypeVar('IterableDataset')
# distribution strategy rename
DistributionStrategy = TypeVar('DistributionStrategy')
# neural network model
Model = TypeVar('Model')
# any property type
Property = TypeVar('Property')
# array like objects
ArrayLike = TypeVar('ArrayLike')
