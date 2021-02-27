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
from kat_api.metadata import IMetaData
from kat_api.network import NetworkInputType
from kat_typing import ArrayLike


class IPropertyDescriptor(IMetaData):
    """
    Common interface for prop descriptors.
    Id, name, type. (can be simple type, or also object)
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
        return (hasattr(subclass, 'get_id') and
                callable(subclass.get_id) and
                hasattr(subclass, 'get_data_type') and
                callable(subclass.get_data_type) or
                NotImplemented)

    @abstractmethod
    def get_id(self) -> str:
        """
        Unique Id.
        The developer is responsible for the uniqueness.

        :return:
            string representation of the uid
        """
        pass


class ITensorDescriptor(IPropertyDescriptor):
    """
    Common interface for tensor descriptors.
    Data type with shape. (np.ndarray like objects)
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
        return (hasattr(subclass, 'get_tensor_shape') and
                callable(subclass.get_tensor_shape) and
                hasattr(subclass, 'get_data_type') and
                callable(subclass.get_data_type) and
                hasattr(subclass, 'get_network_input_type') and
                callable(subclass.get_network_input_type) and
                hasattr(subclass, 'get_low') and
                callable(subclass.get_low) and
                hasattr(subclass, 'get_high') and
                callable(subclass.get_high) or
                NotImplemented)

    @abstractmethod
    def get_tensor_shape(self) -> tuple:
        """
        :returns:
            Tensor dimensions in tuple format.
        """
        pass

    @abstractmethod
    def get_data_type(self) -> type:
        """
        Type of the described data.
        data_type = type(some_object)

        :return:
            type of the described data
        """
        pass

    @abstractmethod
    def get_network_input_type(self) -> NetworkInputType:
        """
        The suggested input type of the tensor.

        :return:
             NetworkInputType enum
        """
        pass

    @abstractmethod
    def get_low(self) -> ArrayLike:
        """
        Returns the lower bounds of the tensor dimensions.

        :returns:
            an array which has an item number of dim(tensor) containing the dimension
            independent lower bound values
        """
        pass

    @abstractmethod
    def get_high(self) -> ArrayLike:
        """
        Returns the higher bounds of the tensor dimensions.

        :returns:
            an array which has an item number of dim(tensor) containing the dimension
            independent higher bound values
        """
        pass
