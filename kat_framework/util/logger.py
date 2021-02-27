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


import logging
import logging.config
from logging import Logger


logging.config.fileConfig(fname='logger.conf', disable_existing_loggers=False)


def get_logger(module_name: str) -> Logger:
    """
    Create a logger instance with the specified module context.

    :param module_name:
        mainly the caller module's name for logging context
    :return:
        logger instance
    """
    return logging.getLogger(module_name)
