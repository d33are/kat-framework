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


from abc import abstractmethod
from kat_api.singleton import SingletonMeta
from enum import Enum
from typing import Type
from kat_typing import Property


class ConfigurationProperty(Enum):
    """
    Abstract base for configuration properties.
    """
    def __new__(cls, label, prop_type, default_value):
        obj = object.__new__(cls)
        obj.label = label
        obj.prop_type = prop_type
        obj.default_value = default_value
        obj._value_ = label
        return obj


class IConfigurationHandler(metaclass=SingletonMeta):
    """
    Common interface for configuration handlers.
    IConfigurationHandlers are responsible for:
        * loading the settings from the specified resource
        * storing the settings (assuming in memory caching)
        * giving access to the settings in a typesafe and thread safe way
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
        return (hasattr(subclass, 'get_config_property') and
                callable(subclass.get_config_property) or
                NotImplemented)

    @abstractmethod
    def get_config_property(
            self, property_descriptor: ConfigurationProperty, expected_type: Type[Property]) -> Property:
        """
        Get a (type safe) property value from the application's configuration, based
        on the given property descriptor.

        :param expected_type:
            expected return type

        :param property_descriptor:
            descriptor of the property

        :return:
            type safe value if present, default value if present or None
        """
        pass
