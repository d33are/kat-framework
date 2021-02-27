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

from kat_framework.networks.base import Network
from kat_framework.core.descriptors import ITensorDescriptor
from kat_typing import TrainLoss, Tensor, Policy, DistributionStrategy
from kat_api import INetwork, IReadOnlyMemory
from typing import Optional
from overrides import overrides
import numpy as np


class RandomActionNetwork(Network, INetwork):
    """
    Random preprocess network for testing purposes.

    This makes this network no matter what the input is, a pseudo random like network.
    """

    def __init__(self, name: str = 'RandomActionNetwork'):
        """
        Default constructor.
        """
        super(RandomActionNetwork, self).__init__(name=name)

    def init(self,
             output_descriptor: ITensorDescriptor,
             replay_memory_access: Optional[IReadOnlyMemory] = None,
             input_descriptor: Optional[ITensorDescriptor] = None,
             is_distribution_enabled: Optional[bool] = False,
             strategy: Optional[DistributionStrategy] = None,
             target_network: Optional = None):
        """
        Object initialization.

        # see : INetwork.init(self,
                        output_descriptor: ITensorDescriptor,
                        replay_memory_access: Optional[IReadOnlyMemory] = None,
                        input_descriptor: Optional[ITensorDescriptor] = None,
                        is_distribution_enabled: Optional[bool] = False,
                        strategy: Optional[DistributionStrategy] = None,
                        target_network: Optional = None)
        """
        super(RandomActionNetwork, self).init(input_descriptor=input_descriptor,
                                              output_descriptor=output_descriptor)

    @overrides
    def train_batch(self, current_episode: Optional[int] = 0, current_step: Optional[int] = 0) -> TrainLoss:
        """
        Object initialization.

        # see : INetwork.train_batch(self, current_episode: Optional[int] = 0, current_step: Optional[int] = 0)
        """
        return np.random.uniform(0.0, 100.0)

    @overrides
    def get_distribution_strategy(self) -> DistributionStrategy:
        """
        # see : INetwork.get_distribution_strategy()
        """
        return None

    @overrides
    def get_weights(self):
        """
        # see : INetwork.get_weights()
        """
        return None

    @overrides
    def set_weights(self, weights: object):
        """
        # see : INetwork.set_weights(weights)
        """
        pass

    # protected member functions

    @overrides
    def _load_configuration(self):
        """
        Loads the configuration.

        # see : Network._load_configuration()
        """
        super(RandomActionNetwork, self)._load_configuration()

    @overrides
    def _predict(self, input_tensor: Tensor) -> Policy:
        """
        Predicts the next action.

        # see : Network._predict(input_tensor)
        """
        return np.empty(self._network_output_descriptor.get_tensor_shape(),
                        self._network_output_descriptor.get_data_type())
