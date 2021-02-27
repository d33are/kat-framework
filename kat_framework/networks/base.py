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

from kat_framework.framework import KatherineApplication
from kat_framework.config.config_props import NetworkConfigurationProperty
from kat_framework.util import logger, tensors
from kat_api import ITensorDescriptor, IModelSerializer, IConfigurationHandler
from kat_typing import Tensor, Policy, Model, TrainLoss
from typing import Optional
from abc import ABCMeta, abstractmethod
from logging import Logger

UNCHECKED_WARN_MSG = "Unchecked input will be fed to the network, assuming unexpected behavior."


class Network(metaclass=ABCMeta):
    """
    Abstract base for networks.

    # see : INetwork
    """

    # protected members

    _log: Logger = None
    _config_handler: IConfigurationHandler = None
    _network_model: Model = None
    _serializer: IModelSerializer = None
    _input_descriptor: ITensorDescriptor = None
    _last_checkpoint: int = 0
    _checkpoint_frequency: int = None
    _name: str = None
    _network_output_descriptor: ITensorDescriptor = None

    # public member functions

    def __init__(self, name: Optional[str] = None):
        """Creates an instance of `Network`.

        Args:
          name: A string representing the name of the network.
        """
        self._log = logger.get_logger(self.__class__.__name__)
        self._config_handler = KatherineApplication.get_application_config()
        self._serializer = KatherineApplication.get_application_factory().build_model_serializer()
        self._name = name or None
        self._load_configuration()

    @property
    def input_descriptor(self):
        """
        Returns the input descriptor property.
        """
        return self._input_descriptor

    def init(self,
             output_descriptor: ITensorDescriptor,
             input_descriptor: Optional[ITensorDescriptor] = None) -> None:
        """
        Object initialization.

        :param output_descriptor:
            network output descriptor
        :param input_descriptor:
            network input descriptor
        """
        self._input_descriptor = input_descriptor
        self._network_output_descriptor = output_descriptor
        if len(self._network_output_descriptor.get_tensor_shape()) > 1:
            raise ValueError("assuming 1 dimensional vector output")

    def predict(self, input_tensor: Tensor) -> Policy:
        """
        It will validates the first argument (`observation`)
        against `self.input_descriptor` if one is available.

        # see : INetwork.predict(input_tensor)

        :param input_tensor:
            The input to `self.call`, matching `self.input_descriptor`.
        :returns
          updated policy as `IPolicy` and the predicted action for the given state
        """
        if self.input_descriptor is not None:
            if not tensors.check_same_tensor_structure(input_tensor, self.input_descriptor, reduce_batch_dim=True):
                raise RuntimeError("Input structure vs observation mismatch.")
        else:
            self._log.warning(UNCHECKED_WARN_MSG)
        return self._predict(input_tensor)

    def train_batch(self, current_episode: Optional[int] = 0, current_step: Optional[int] = 0) -> TrainLoss:
        """
        # see: INetwork.train_batch(current_episode: Optional[int] = 0, current_step: Optional[int] = 0)
        """
        if current_episode % self._checkpoint_frequency == 0 and self._last_checkpoint != current_episode:
            self._last_checkpoint = current_episode
            self._serializer.save_checkpoint(self._network_model, self._name)
        return 0

    def persist_model(self):
        """
        # see: INetwork.persist_model()
        """
        if self._network_model is not None:
            self._serializer.save_model(self._network_model, self._name)

    # protected member functions

    @abstractmethod
    def _predict(self, input_tensor: Tensor) -> Policy:
        """
        Abstract predict method for derived classes.
        """
        pass

    def _load_configuration(self):
        """
        Loads necessary configurations.
        """
        self._checkpoint_frequency = self._config_handler.get_config_property(
            NetworkConfigurationProperty.CHECKPOINT_FREQUENCY,
            NetworkConfigurationProperty.CHECKPOINT_FREQUENCY.prop_type)
