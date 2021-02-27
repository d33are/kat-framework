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

from kat_typing import Model
from kat_api import IModelStorageDriver
from overrides import overrides


class DummyStorageDriver(IModelStorageDriver):
    """
    Dummy storage driver for testing purposes.

    This class mainly does nothing.

    # see : IModelStorageDriver
    """

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        pass

    @overrides
    def save(self, model: Model, filepath: str):
        """
        # see : IModelStorageDriver.save(model, filepath)
        """
        if model is None:
            raise ValueError("No model specified.")
        if filepath is None:
            raise ValueError("No path specified.")

    @overrides
    def restore(self, filepath: str) -> Model:
        """
        # see : IModelStorageDriver.restore(filepath)
        """
        if filepath is None:
            raise ValueError("No path specified.")
        return None

    @overrides
    def save_checkpoint(self, model: Model, filepath: str) -> None:
        """
        # see : IModelStorageDriver.save_checkpoint(model, filepath)
        """
        if model is None:
            raise ValueError("No model specified.")
        if filepath is None:
            raise ValueError("No path specified.")

    @overrides
    def restore_checkpoint(self, model: Model, filepath: str) -> Model:
        """
        # see : IModelStorageDriver.restore_checkpoint(model, filepath)
        """
        if model is None:
            raise ValueError("No model specified.")
        if filepath is None:
            raise ValueError("No path specified.")
        return model
