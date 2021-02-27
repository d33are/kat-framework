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
from kat_framework.config.config_props import KatConfigurationProperty
from typing import Optional
import os

MODEL_DIRECTORY_POSTFIX = "models"
METRICS_DIRECTORY_POSTFIX = "metrics"


def build_global_work_directory() -> str:
    """
    Helper function for building the application's global working directory.

    base_work_directory/game/agent/run_tag

    :returns
        the global work directory path (relative path)
    """
    base_work_directory = KatherineApplication.get_application_config().get_config_property(
        KatConfigurationProperty.WORK_DIRECTORY, KatConfigurationProperty.WORK_DIRECTORY.prop_type)
    agent_name = KatherineApplication.get_application_config().get_config_property(
        KatConfigurationProperty.AGENT_CLASS, KatConfigurationProperty.AGENT_CLASS.prop_type)
    game_name = KatherineApplication.get_application_config().get_config_property(
        KatConfigurationProperty.GAME_CLASS, KatConfigurationProperty.GAME_CLASS.prop_type)
    run_tag = KatherineApplication.get_application_config().get_config_property(
        KatConfigurationProperty.RUN_TAG, KatConfigurationProperty.RUN_TAG.prop_type)
    agent_name = agent_name[agent_name.rindex('.') + 1:]
    game_name = game_name[game_name.rindex('.') + 1:]
    return os.path.join(base_work_directory, game_name, agent_name, run_tag)


def build_work_directory(postfix: str) -> str:
    """
    Builds a specific working directory.

    :param postfix:
        the name of the working directory
    :return:
        working dir: base_work_directory/game/agent/run_tag/postfix
    """
    if postfix is None or "" == postfix:
        raise ValueError("No postfix specified.")
    return os.path.join(build_global_work_directory(), postfix)


def build_models_work_directory(models_postfix: Optional[str]) -> str:
    """
    Helper function for building model serialization work directory.
    """
    if models_postfix is None or "" == models_postfix:
        models_postfix = MODEL_DIRECTORY_POSTFIX
    return build_work_directory(models_postfix)


def build_metrics_work_directory(metrics_postfix: Optional[str]) -> str:
    """
    Helper function for building metrics work directory.
    """
    if metrics_postfix is None or "" == metrics_postfix:
        metrics_postfix = METRICS_DIRECTORY_POSTFIX
    return build_work_directory(metrics_postfix)
