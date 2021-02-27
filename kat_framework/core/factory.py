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
from kat_framework.util import reflection
from kat_framework.config.config_props import KatConfigurationProperty
from kat_framework.memory.access import MemoryAccessor
from kat_api import IFactory, IGame, IDriver, IAgent, INetwork, IReplayMemory, IMetricTracer
from kat_api import IModelSerializer, IModelStorageDriver, IReadOnlyMemory
from overrides import overrides


class KatFactory(IFactory):
    """
    Default application scoped factory for the reference implementation.
    """

    def __init__(self):
        """
        Default constructor.
        """
        self._config_handler = KatherineApplication.get_application_config()

    @overrides
    def build_game(self) -> IGame:
        """
        # see : IFactory.build_game()
        """
        game_class_string = str(
            self._config_handler.get_config_property(KatConfigurationProperty.GAME_CLASS,
                                                     KatConfigurationProperty.GAME_CLASS.prop_type))
        return reflection.get_instance(game_class_string, IGame)

    @overrides
    def build_driver(self) -> IDriver:
        """
        # see : IFactory.build_driver()
        """
        driver_class_string = str(
            self._config_handler.get_config_property(KatConfigurationProperty.DRIVER_CLASS,
                                                     KatConfigurationProperty.DRIVER_CLASS.prop_type))
        return reflection.get_instance(driver_class_string, IDriver)

    @overrides
    def build_agent(self) -> IAgent:
        """
        # see : IFactory.build_agent()
        """
        agent_class_string = str(
            self._config_handler.get_config_property(KatConfigurationProperty.AGENT_CLASS,
                                                     KatConfigurationProperty.AGENT_CLASS.prop_type))
        return reflection.get_instance(agent_class_string, IAgent)

    @overrides
    def build_network(self) -> INetwork:
        """
        # see : IFactory.build_network()
        """
        network_class_string = str(
            self._config_handler.get_config_property(KatConfigurationProperty.NETWORK_CLASS,
                                                     KatConfigurationProperty.NETWORK_CLASS.prop_type))
        return reflection.get_instance(network_class_string, INetwork)

    @overrides
    def build_memory(self) -> IReplayMemory:
        """
        # see : IFactory.build_memory()
        """
        memory_class_string = str(
            self._config_handler.get_config_property(KatConfigurationProperty.MEMORY_CLASS,
                                                     KatConfigurationProperty.MEMORY_CLASS.prop_type))
        return reflection.get_instance(memory_class_string, IReplayMemory)

    @overrides
    def build_memory_access(self) -> IReadOnlyMemory:
        """
        # see : IFactory.build_memory_access()
        """
        return MemoryAccessor()

    @overrides
    def build_metrics_tracer(self) -> IMetricTracer:
        """
        # see : IFactory.build_metrics_tracer()
        """
        tracer_class_string = str(
            self._config_handler.get_config_property(KatConfigurationProperty.METRICS_TRACER_CLASS,
                                                     KatConfigurationProperty.METRICS_TRACER_CLASS.prop_type))
        return reflection.get_instance(tracer_class_string, IMetricTracer)

    @overrides
    def build_model_serializer(self) -> IModelSerializer:
        """
        # see : IFactory.build_model_serializer()
        """
        model_serializer_class = str(
            self._config_handler.get_config_property(KatConfigurationProperty.MODEL_SERIALIZER_CLASS,
                                                     KatConfigurationProperty.MODEL_SERIALIZER_CLASS.prop_type))
        return reflection.get_instance(model_serializer_class, IModelSerializer)

    @overrides
    def build_model_storage_driver(self) -> IModelStorageDriver:
        """
        # see : IFactory.build_model_storage_driver()
        """
        model_storage_driver_class = str(
            self._config_handler.get_config_property(KatConfigurationProperty.MODEL_STORAGE_DRIVER_CLASS,
                                                     KatConfigurationProperty.MODEL_STORAGE_DRIVER_CLASS.prop_type))
        return reflection.get_instance(model_storage_driver_class, IModelStorageDriver)
