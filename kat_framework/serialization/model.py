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
from kat_framework.config.config_props import ModelSerializerProperty
from kat_framework.util import fileio
from kat_typing import Model
from kat_api import IModelSerializer, IModelStorageDriver
from overrides import overrides
import os


WORKING_DIRECTORY_PREFIX = "models"
MODEL_DIRECTORY_PREFIX = "saved_model"
CHECKPOINT_DIRECTORY_PREFIX = "checkpoints"


class KatModelSerializer(IModelSerializer):
    """
    Default model serializer implementation.

    # see : IModelSerializer
    """

    # protected members

    _is_model_persistence_enabled: bool = False
    _is_checkpoints_enabled: bool = False
    _work_directory: str = None
    _storage_driver: IModelStorageDriver = None

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        self._load_configuration()
        self._work_directory = fileio.build_models_work_directory(WORKING_DIRECTORY_PREFIX)
        self._storage_driver = KatherineApplication.get_application_factory().build_model_storage_driver()

    @overrides
    def save_model(self, model: Model, model_name: str) -> None:
        """
        # see : IModelSerializer.save_model(model, model_name)
        """
        if not self._is_model_persistence_enabled:
            return
        self._storage_driver.save(
            model, os.path.join(self._work_directory, MODEL_DIRECTORY_PREFIX, model_name))

    @overrides
    def restore_model(self, filepath: str) -> Model:
        """
        # see : IModelSerializer.restore_model(filepath)
        """
        return self._storage_driver.restore(filepath)

    @overrides
    def save_checkpoint(self, model: Model, model_name: str) -> None:
        """
        # see : IModelSerializer.save_checkpoint(model, model_name)
        """
        if not self._is_checkpoints_enabled:
            return
        self._storage_driver.save_checkpoint(
            model, os.path.join(self._work_directory, CHECKPOINT_DIRECTORY_PREFIX, model_name))

    @overrides
    def restore_checkpoint(self, model: Model, filepath: str) -> Model:
        """
        # see : IModelSerializer.restore_checkpoint(model, filepath)
        """
        return self._storage_driver.restore_checkpoint(model, filepath)

    # protected member functions

    def _load_configuration(self) -> None:
        """
        Loads the configuration.
        """
        config_handler = KatherineApplication.get_application_config()
        self._is_model_persistence_enabled = config_handler.get_config_property(
            ModelSerializerProperty.MODEL_PERSISTENCE_ENABLED,
            ModelSerializerProperty.MODEL_PERSISTENCE_ENABLED.prop_type)
        self._is_checkpoints_enabled = config_handler.get_config_property(
            ModelSerializerProperty.MODEL_CHECKPOINTS_ENABLED,
            ModelSerializerProperty.MODEL_CHECKPOINTS_ENABLED.prop_type)
