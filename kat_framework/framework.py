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


from kat_api import IFactory
from kat_api import IConfigurationHandler
from kat_framework.util import reflection, logger

LOGO_STRING = \
    "\n ##################################################\n \
#  _   __      _   _               _             #\n \
# | | / /     | | | |             (_)            #\n \
# | |/ /  __ _| |_| |__   ___ _ __ _ _ __   ___  #\n \
# |    \\ / _` | __| '_ \\ / _ \\ '__| | '_ \\ / _ \\ #\n \
# | |\\  \\ (_| | |_| | | |  __/ |  | | | | |  __/ #\n \
# \\_| \\_/\\__,_|\\__|_| |_|\\___|_|  |_|_| |_|\\___| #\n \
#                                                #\n \
# General Video Game AI                          #\n \
# Copyright (C) 2020-2021 d33are                 #\n \
##################################################\n"

log = logger.get_logger(__name__ + ".KatherineApplication")


class KatherineApplication:
    """
    Katherine application entrypoint.
    """
    application_factory: IFactory
    config_handler: IConfigurationHandler

    @staticmethod
    def init(factory_class: str, config_handler_class: str, config_uri: str) -> None:
        """
        Runtime initialization.

        :param factory_class:
            factory implementation's  canonical name
        :param config_handler_class:
            config implementation's  canonical name
        :param config_uri:
            configuration uri
        """
        if not factory_class:
            raise ValueError("No factory specified.")
        if not config_handler_class:
            raise ValueError("No config handler specified.")
        if not config_uri:
            raise ValueError("No config_uri specified.")
        KatherineApplication.config_handler = reflection.get_instance(
            config_handler_class, IConfigurationHandler, config_uri)
        KatherineApplication.application_factory = reflection.get_instance(factory_class, IFactory)

    @staticmethod
    def run(factory_class: str, config_handler_class: str, config_uri: str, enable_logo: bool = True) -> None:
        """
        Runs the main loop.

        :param factory_class:
            factory implementation's  canonical name
        :param config_handler_class:
            config implementation's  canonical name
        :param config_uri:
            configuration uri
        :param enable_logo:
            display the logo in the logs, or not
        """
        if enable_logo:
            log.info(LOGO_STRING)
        KatherineApplication.init(factory_class, config_handler_class, config_uri)
        driver = KatherineApplication.get_application_factory().build_driver()
        driver.run()

    @staticmethod
    def get_application_factory() -> IFactory:
        """
        Returns the application scoped factory object if exists.

        :return:
            `IFactory` instance, or None
        """
        return KatherineApplication.application_factory

    @staticmethod
    def get_application_config() -> IConfigurationHandler:
        """
        Returns the application scoped config object if exists.

        :return:
            `IConfigurationHandler` instance, or None
        """
        return KatherineApplication.config_handler
