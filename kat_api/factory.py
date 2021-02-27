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
from kat_api.game import IGame
from kat_api.driver import IDriver
from kat_api.agent import IAgent
from kat_api.network import INetwork
from kat_api.memory import IReplayMemory, IReadOnlyMemory
from kat_api.metrics import IMetricTracer
from kat_api.model import IModelSerializer, IModelStorageDriver


class IFactory(metaclass=ABCMeta):
    """
    Factory interface for the framework.

    The factory implementation needs to implement some object instantiation, which objects are
    mandatory for the framework.
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
        return (hasattr(subclass, 'build_game') and
                callable(subclass.build_game) and
                hasattr(subclass, 'build_driver') and
                callable(subclass.build_driver) and
                hasattr(subclass, 'build_agent') and
                callable(subclass.build_agent) and
                hasattr(subclass, 'build_network') and
                callable(subclass.build_network) and
                hasattr(subclass, 'build_memory') and
                callable(subclass.build_memory) and
                hasattr(subclass, 'build_memory_access') and
                callable(subclass.build_memory_access) and
                hasattr(subclass, 'build_metrics_tracer') and
                callable(subclass.build_metrics_tracer) and
                hasattr(subclass, 'build_model_serializer') and
                callable(subclass.build_model_serializer) and
                hasattr(subclass, 'build_model_storage_driver') and
                callable(subclass.build_model_storage_driver) or
                NotImplemented)

    @abstractmethod
    def build_game(self) -> IGame:
        """
        Building the game wrapper for the framework.

        :return:
            a proper game instance
        """
        pass

    @abstractmethod
    def build_driver(self) -> IDriver:
        """
        Building the driver object for the framework.

        :return:
            a proper driver instance
        """
        pass

    @abstractmethod
    def build_agent(self) -> IAgent:
        """
        Building the agent object for the framework.

        :return:
            a proper agent instance
        """
        pass

    @abstractmethod
    def build_network(self) -> INetwork:
        """
        Building the network object for the agent.

        :return:
            a proper network instance
        """
        pass

    @abstractmethod
    def build_memory(self) -> IReplayMemory:
        """
        Building the replay object for the agent.

        :return:
            a proper memory instance
        """
        pass

    @abstractmethod
    def build_memory_access(self) -> IReadOnlyMemory:
        """
        Building the replay memory access object for the agent.

        :return:
            a proper memory instance
        """
        pass

    @abstractmethod
    def build_metrics_tracer(self) -> IMetricTracer:
        """
        Building the metrics object for the driver.

        :return:
            a proper metrics instance
        """
        pass

    @abstractmethod
    def build_model_serializer(self) -> IModelSerializer:
        """
        Building model serializer object.

        :returns:
            a proper serializer instance
        """
        pass

    @abstractmethod
    def build_model_storage_driver(self) -> IModelStorageDriver:
        """
        Building model storage driver object.

        :returns:
            a proper storage driver instance
        """
        pass
