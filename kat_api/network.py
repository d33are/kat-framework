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


from abc import ABCMeta, abstractmethod
from kat_typing import Tensor, Policy, TrainLoss, DistributionStrategy
from kat_api.prop_desc import IMetaData
from kat_api.memory import IReadOnlyMemory
from typing import Optional
from enum import Enum


class NetworkInputType(Enum):
    """
    Input type flag for easy processing.
    """
    IMG = "img"
    RAM = "ram"
    NONE = "none"


class INetwork(metaclass=ABCMeta):
    """
    Common interface for networks.

    Network implementations are mostly encapsulates the concrete AI model.
    Keras, Tensorflow, Pytorch or any other framework implementations
    in a pluggable and/or modular manner.

    This interface is designed for making a transparent behavior between any A.I.
    frameworks and agent implementations.
    """

    @classmethod
    def __subclasshook__(cls, subclass: object):
        """Checks the class' expected behavior as a formal python interface.

        :param subclass:
            class to be checked

        :return:
            True if the object is a "real implementation" of this interface,
            otherwise False.
        """
        return (hasattr(subclass, 'init') and
                callable(subclass.init) and
                hasattr(subclass, 'train_batch') and
                callable(subclass.train_batch) and
                hasattr(subclass, 'predict') and
                callable(subclass.predict) and
                hasattr(subclass, 'get_distribution_strategy') and
                callable(subclass.get_distribution_strategy) and
                hasattr(subclass, 'persist_model') and
                callable(subclass.persist_model) or
                NotImplemented)

    @abstractmethod
    def init(self,
             output_descriptor: IMetaData,
             replay_memory_access: Optional[IReadOnlyMemory] = None,
             input_descriptor: Optional[IMetaData] = None,
             is_distribution_enabled: Optional[bool] = False,
             strategy: Optional[DistributionStrategy] = None,
             target_network: Optional = None) -> None:
        """
        We're assuming that Networks gonna' be instantiated through factory interfaces. Based on that
        knowledge, an ideal Network implementation has a "no args" constructor, and a separated initialization
        method. (object instantiation is _NOT_ initialization)

        :param output_descriptor:
            tensor descriptor for the model output (fc layer dimension)
        :param is_distribution_enabled:
            (Optional) distributed learning is enabled or not
        :param replay_memory_access:
            (Optional) memory accessor for experience replay based networks
        :param input_descriptor:
            (Optional) tensor descriptor for the model input layer. Networks are able to define
            their own input descriptor, in this case, descriptor is not necessary to be provided
        :param strategy
            (Optional) Modern ML APIs can distribute training across multiple GPUs, multiple machines. It means
            the implementation must able to "know" that it is training on a single machine or on a cluster.

        :param target_network (Optional)
            External target network for evaluation, if necessary
        """
        pass

    @abstractmethod
    def train_batch(self, current_episode: Optional[int] = 0, current_step: Optional[int] = 0) -> TrainLoss:
        """
        Trains the network for optimal environment interactions. This Network interface is designed for
        "episodic step" drivers, so we're assuming to do one train step per call.

        :returns
            Current loss value after executing the train step, provided by the loss function.
        """
        pass

    @abstractmethod
    def predict(self, input_tensor: Tensor) -> Policy:
        """
        Generates output prediction from the input observation.

        A typical `call` method in an `INetwork` implementation will have a
        signature that accepts `inputs`, as well as other `*args` and `**kwargs`.

        :param input_tensor:
            The input to `self.call`, matching `self.input_descriptor`.
        :returns:
            predicted policy to use
        """
        pass

    @abstractmethod
    def get_distribution_strategy(self) -> DistributionStrategy:
        """
        Modern ML APIs can distribute training across multiple GPUs, multiple machines. It means
        the implementation must able to "know" that it is training on a single machine or on a cluster.

        :returns:
            the agent's distribution strategy
        """
        pass

    @abstractmethod
    def persist_model(self) -> None:
        """
        Assuming, the Network is able to persist its policy or network in a stateful manner.
        Implementations must use `IModelSerializer`.
        This can be:
            * saving policy tables
            * saving model weights
            * saving whole model graphs
            ... and so on
        """
        pass

    @abstractmethod
    def get_weights(self) -> object:
        """
        :returns
            the current weight values of the network.
        """
        pass

    @abstractmethod
    def set_weights(self, weights: object):
        """
        Sets the specified weights to the network.
        """
        pass
