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

from kat_api import IMetaData, NetworkInputType, IPropertyDescriptor, ITensorDescriptor
from kat_typing import ArrayLike
from overrides import overrides
from typing import Optional
import uuid
import numpy as np


class Metadata(IMetaData):
    """
    Abstract base class for metadata based implementations.

    Metadata "means" class or property descriptors in most of the framework.
    """

    # protected members

    _display_name: str = None

    # public member functions

    def __init__(self, display_name: str):
        """
        Default constructor.

        :param display_name:
            display name of the metadata
        """
        self._display_name = display_name or uuid.uuid1()

    def __eq__(self, other):
        if isinstance(other, Metadata):
            return self.get_display_name() == other.get_display_name()
        return False

    @overrides
    def get_display_name(self) -> str:
        """
        Returns the display name property.
        """
        return self._display_name


class PropertyDescriptor(Metadata, IPropertyDescriptor):
    """
    Describes a property of a class.

    Id, name, type. (can be simple type, or also object)
    """

    # protected members

    _identifier: str = None

    # public member functions

    def __init__(self, display_name: str, identifier: Optional[str]):
        """
        Default constructor.

        :param display_name:
            display name of the metadata
        :param identifier:
            unique identifier of the metadata
        """
        super().__init__(display_name)
        self._identifier = identifier or uuid.uuid1()

    def __eq__(self, other):
        if isinstance(other, PropertyDescriptor):
            return super(PropertyDescriptor, self).__eq__(other) and self.get_id() == other.get_id()
        return False

    @overrides
    def get_id(self) -> str:
        """
        Returns the unique identifier.
        """
        return self._identifier


class TensorDescriptor(PropertyDescriptor, ITensorDescriptor):
    """
    Describes Tensor data, dtype with shape.

    # see : `ITensorDescriptor
    """

    # protected members

    _data_type: type = None
    _input_type: NetworkInputType = None
    _tensor_shape: tuple = None
    _low: object = None
    _high: object = None

    # public member functions

    def __init__(self,
                 display_name: str,
                 data_type: type,
                 tensor_shape: tuple,
                 input_type: Optional[NetworkInputType] = None,
                 identifier: Optional[str] = None,
                 low: Optional[ArrayLike] = None,
                 high: Optional[ArrayLike] = None):
        """
        Default constructor.

        :param display_name
            display name of the metadata
        :param identifier:
            unique identifier of the metadata
        :param data_type:
            tensor's datatype
        :param tensor_shape:
            tensor shape
        :param input_type:
            `NetworkInputType`
        :param low:
            array like boundary definitions (low)
        :param high:
            array like boundary definitions (high)
        """
        super().__init__(display_name, identifier)
        if not data_type:
            raise ValueError("No data type specified")
        if tensor_shape is None:
            raise ValueError("No tensor shape specified")
        tensor_shape = tuple(tensor_shape)
        if low is not None and not np.isscalar(low) and not low.shape == tensor_shape:
            raise ValueError("low.shape doesn't match provided tensor_shape")
        if high is not None and not np.isscalar(high) and not high.shape == tensor_shape:
            raise ValueError("high.shape doesn't match provided tensor_shape")
        if low is not None and np.isscalar(low):
            low = np.full(tensor_shape, low, dtype=data_type)
        if high is not None and np.isscalar(high):
            high = np.full(tensor_shape, high, dtype=data_type)
        self._data_type = data_type
        self._input_type = input_type
        self._tensor_shape = tensor_shape or ()
        self._low = low
        self._high = high

    @overrides
    def get_data_type(self) -> type:
        """
        # see: ITensorDescriptor.get_data_type()
        """
        return self._data_type

    @overrides
    def get_network_input_type(self) -> NetworkInputType:
        """
        # see: ITensorDescriptor.get_network_input_type()
        """
        return self._input_type

    @overrides
    def get_tensor_shape(self) -> tuple:
        """
        # see: ITensorDescriptor.get_tensor_shape()
        """
        return self._tensor_shape

    @overrides
    def get_low(self) -> ArrayLike:
        """
        # see: ITensorDescriptor.get_low()
        """
        return self._low

    @overrides
    def get_high(self) -> ArrayLike:
        """
        # see: ITensorDescriptor.get_high()
        """
        return self._high
