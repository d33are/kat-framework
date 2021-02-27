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

import tensorflow as tf
from kat_typing import Model
from kat_api import IModelStorageDriver
import uuid


class TensorflowStorageDriver(IModelStorageDriver):
    """
    Tensorflow based storage driver implementation.

    # see : IModelStorageDriver
    """

    def __init__(self):
        pass

    def save(self, model: Model, filepath: str):
        """
        # see : IModelStorageDriver.save(self, model: Model, filepath: str)
        """
        if model is None:
            raise ValueError("No model specified.")
        if filepath is None:
            raise ValueError("No path specified.")
        tf.keras.models.save_model(model=model, filepath=filepath)

    def restore(self, filepath: str) -> Model:
        """
        # see : IModelStorageDriver.restore(self, filepath: str)
        """
        if filepath is None:
            raise ValueError("No path specified.")
        return tf.keras.models.load_model(filepath=filepath)

    def save_checkpoint(self, model: Model, filepath: str) -> None:
        """
        # see : IModelStorageDriver.save_checkpoint(self, model: Model, filepath: str)
        """
        if model is None:
            raise ValueError("No model specified.")
        if filepath is None:
            raise ValueError("No path specified.")
        if not isinstance(model, tf.keras.Model):
            raise TypeError("Not a tensorflow model.")
        filepath += "_" + str(uuid.uuid1())
        model.save_weights(filepath=filepath)

    def restore_checkpoint(self, model: Model, filepath: str) -> Model:
        """
        # see : IModelStorageDriver.restore_checkpoint(self, model: Model, filepath: str)
        """
        if model is None:
            raise ValueError("No model specified.")
        if filepath is None:
            raise ValueError("No path specified.")
        if not isinstance(model, tf.keras.Model):
            raise TypeError("Not a tensorflow model.")
        model.load_weights(filepath=filepath)
        return model
