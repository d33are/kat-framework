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
from kat_typing import Model


class IModelSerializer(metaclass=ABCMeta):
    """
    Common interface for Model serializer implementations.

    Model serializers are responsible for transforming neural network models into a
    "reversible" format. Assuming that serializers are able to save the entire model, or
    just the model's weights. (full save, checkpointing)

    Implementations should use `IModelStorageDriver` implementations for persisting model instances,
    but in simple cases it is not necessary.
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
        return (hasattr(subclass, 'save_model') and
                callable(subclass.save_model) and
                hasattr(subclass, 'restore_model') and
                callable(subclass.restore_model) and
                hasattr(subclass, 'save_checkpoint') and
                callable(subclass.save_checkpoint) and
                hasattr(subclass, 'restore_checkpoint') and
                callable(subclass.restore_checkpoint) or
                NotImplemented)

    @abstractmethod
    def save_model(self, model: Model, model_name: str) -> None:
        """
        Persists the entire model with the specified name. We're assuming that
        the implementation will handles the working directory.

        :param model:
            model to save
        :param model_name:
            name of the model
        """
        pass

    @abstractmethod
    def restore_model(self, filepath: str) -> Model:
        """
        Restores a model from the specified file path.

        :param filepath:
            model's path on any persistent storage (uri)
        :returns:
            deserialized model instance
        """
        pass

    @abstractmethod
    def save_checkpoint(self, model: Model, model_name: str) -> None:
        """
        Persists the weights of the model with the specified name. We're assuming that
        the implementation will handles the working directory.

        :param model:
            model to save
        :param model_name:
            name of the model
        """

    @abstractmethod
    def restore_checkpoint(self, model: Model, filepath: str) -> Model:
        """
        Restores a checkpoint from the specified file path.

        :param model:
            provided model for restoring weights
        :param filepath:
            checkpoint's path on any persistent storage (uri)
        :returns:
            the specified model with the restored and updated weights
        """
        pass


class IModelStorageDriver(metaclass=ABCMeta):
    """
    Common interface for IStorageDriver implementations.

    Abstraction layer to separate serialization business logic from the exact
    persistence logic.

    For example it can be a filesystem driver, database driver, or framework specific
    model serialization implementations. (Tensorflow, PyTorch)
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
        return (hasattr(subclass, 'save') and
                callable(subclass.save) and
                hasattr(subclass, 'restore') and
                callable(subclass.restore) and
                hasattr(subclass, 'save_checkpoint') and
                callable(subclass.save_checkpoint) and
                hasattr(subclass, 'restore_checkpoint') and
                callable(subclass.restore_checkpoint) or
                NotImplemented)

    @abstractmethod
    def save(self, model: Model, filepath: str):
        """
        Persists the entire model to the specified path.

        :param model:
            model to save
        :param filepath:
            saving path (filesystem path, or database path, etc...)
        """
        pass

    @abstractmethod
    def restore(self, filepath: str) -> Model:
        """
        Restores a model from the specified file path.

        :param filepath:
            model's path on any persistent storage (uri)
        :returns:
            deserialized model instance
        """
        pass

    @abstractmethod
    def save_checkpoint(self, model: Model, filepath: str) -> None:
        """
        Persists the weights of the model to the specified path.

        :param model:
            model to save
        :param filepath:
            saving path (filesystem path, or database path, etc...)
        """
        pass

    @abstractmethod
    def restore_checkpoint(self, model: Model, filepath: str) -> Model:
        """
        Restores a checkpoint from the specified file path.

        :param model:
            provided model for restoring weights
        :param filepath:
            checkpoint's path on any persistent storage (uri)
        :returns:
            the specified model with the restored and updated weights
        """
        pass
