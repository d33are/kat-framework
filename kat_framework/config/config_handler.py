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

from kat_api import IConfigurationHandler, ConfigurationProperty
from kat_framework.util import logger
from kat_typing import Property
from overrides import overrides
from typing import Type
from himl import ConfigProcessor
from logging import Logger
import uritools
import sys

URI_ERROR_MSG = "Wrong URI format"
DUPLICATE_KEY_ERROR_MSG = "Duplicate key in configuration file: %s"
OS_ERROR_MSG = "Configuration file doesn't exists: %s"
PROPERTY_ERROR_MSG = "No property descriptor specified."
DYNAMIC_CAST_ERROR_MSG = "Dynamic cast error: %s, %s"
ATTRIBUTE_ERROR_MSG = "Attribute %s not found on %s class."
DEV_LOG = False  # lower than debug level


class YamlConfigHandler(IConfigurationHandler):
    """
    YAML file based configuration handler implementation.
    """

    # protected members

    _log: Logger = None

    # public member functions

    def __init__(self, config_uri: str):
        """
        Default constructor.

        :param config_uri:
            RFC-3986 Uniform Resource Identifier (URI)
        """
        self._log = logger.get_logger(self.__class__.__name__)
        self.yaml = ConfigProcessor()
        self.settings = {}
        self._load_configuration(config_uri)

    @overrides
    def get_config_property(
            self, property_descriptor: ConfigurationProperty, expected_type: Type[Property]) -> Property:
        """
        Returns the config value of the specified property with the given type.

        :param property_descriptor:
            the descriptor of the wanted property
        :param expected_type:
            expected return type
        :return:
            the configured value of the setting, or the default value if not present
        """
        if not property_descriptor:
            self._log.error(PROPERTY_ERROR_MSG)
            raise ValueError
        item = self._recursive_search(property_descriptor, self.settings)
        # if no hit at all, then default value
        if item is None:
            return property_descriptor.default_value
        return item

    # protected member functions

    def _load_configuration(self, uri: str) -> None:
        """
        Loads in the configuration from the specified file resource.

        #see `adobe himl` package for more information:
        https://github.com/adobe/himl

        :param uri:
            RFC-3986 Uniform Resource Identifier (URI)
        """
        if not uritools.isuri(uri):
            raise ValueError(URI_ERROR_MSG)
        uri_tuple = uritools.urisplit(uri)
        yaml_file = uri_tuple[2][1:]
        try:
            self.settings = dict(self.yaml.process(path=yaml_file, output_format="yaml", print_data=True))
        except OSError:
            self._log.critical(OS_ERROR_MSG, yaml_file, exc_info=True)
            sys.exit()

    def _recursive_search(self, property_descriptor: ConfigurationProperty, property_settings: dict) -> object:
        """
        Search a key recursively in the given dict. Currently the recursion is only works for dict nested types.
        Early "None" returns are covered by the code. Nested Lists are treated as enums or evaluated as tuples.

        :param property_descriptor:
            descriptor of the property
        :param property_settings:
            dict where we want to search for the property
        :return:
            type casted value, or None
        """
        # hit
        if property_descriptor.label in property_settings:
            item = property_settings[property_descriptor.label]
            if isinstance(item, list):  # yaml parser making lists from enums
                typed_list = []
                if issubclass(property_descriptor.prop_type, tuple):
                    [typed_list.append(eval(x)) for x in item]
                else:
                    try:
                        # if not tuple, then we assuming an enum type
                        [typed_list.append(getattr(property_descriptor.prop_type, x)) for x in item]
                    except (AttributeError, TypeError):
                        if DEV_LOG:
                            self._log.debug(ATTRIBUTE_ERROR_MSG, item, property_descriptor.prop_type)
                return typed_list
            try:
                # trying to resolve as a simple enum type
                return getattr(property_descriptor.prop_type, item)
            except (AttributeError, TypeError):
                if DEV_LOG:
                    self._log.debug(ATTRIBUTE_ERROR_MSG, item, property_descriptor.prop_type)
            try:
                # trying to dynamic cast
                return property_descriptor.prop_type(item)
            except (TypeError, ValueError):
                self._log.error(DYNAMIC_CAST_ERROR_MSG, property_descriptor.label, property_descriptor.prop_type,
                                exc_info=True)
                return property_descriptor.default_value
        # no hit, then recursion (if nested structure is exists)
        for key, value in property_settings.items():
            if isinstance(value, dict):  # nested types are usually dicts
                item = self._recursive_search(property_descriptor, value)
                # covering early None returns
                if item is not None:
                    return item
