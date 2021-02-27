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

from kat_api import IPropertyDescriptor, ITensorDescriptor, IObservation
from kat_typing import Tensor
from kat_framework.core.descriptors import TensorDescriptor
from kat_framework.util import logger
from typing import Collection
import numpy as np

log = logger.get_logger(__name__)


def check_same_tensor_structure(
        input_tensor: Tensor, property_desc: ITensorDescriptor, reduce_batch_dim: bool = False) -> bool:
    """
    Checks that the specified tensor is compatible or not with the provided descriptor.

    :param input_tensor:
        tensor to check
    :param property_desc:
        descriptor checked by
    :param reduce_batch_dim:
        batch dimension is needed or not
    :return:
        true if the tensor have made based on the provided descriptor, else false
    """
    if input_tensor is None:
        raise ValueError("No observation specified or wrong type.")
    if property_desc is None:
        raise ValueError("No descriptor specified or wrong type.")
    if reduce_batch_dim:
        r_input_shape = reduce_tensor_batch_dimension(input_tensor)
    else:
        r_input_shape = input_tensor.shape
    if r_input_shape != property_desc.get_tensor_shape():
        log.debug("Tensor shape : %s -> %s vs %s", str(np.shape(input_tensor)),
                  str(r_input_shape),
                  property_desc.get_tensor_shape())
        return False
    if input_tensor.dtype != property_desc.get_data_type():
        log.debug("Tensor dtype : %s vs %s", str(input_tensor.dtype),
                  property_desc.get_data_type())
        return False
    return True


def check_same_observation_structure(
        observation: IObservation, descriptor: Collection[ITensorDescriptor]) -> bool:
    """
    Checks that the specified observation is compatible or not with the provided descriptors.

    :param observation:
        observation to check
    :param descriptor:
        descriptors checked by
    :returns
        true if the observation have made based one of the provided descriptors, else false
    """
    if observation is None or not isinstance(observation, IObservation):
        raise ValueError("No observation specified or wrong type.")
    if descriptor is None:
        raise ValueError("No descriptor specified or wrong type.")
    for property_desc in descriptor:
        if isinstance(property_desc, ITensorDescriptor):
            property_desc: ITensorDescriptor
            tensor = getattr(observation, property_desc.get_display_name())
            match = check_same_tensor_structure(tensor, property_desc)
            if not match:
                return False
        elif isinstance(property_desc, IPropertyDescriptor):
            log.warning("Descriptor type is not supported: %s", type(property_desc))
    return True


def reduce_spec_batch_dimension(input_spec: ITensorDescriptor) -> ITensorDescriptor:
    """
    Helper function to reduce descriptor batch dimension.
    """
    if input_spec is None:
        raise ValueError("No tensor spec specified.")
    return TensorDescriptor(input_spec.get_display_name(),
                            input_spec.get_data_type(),
                            input_spec.get_tensor_shape()[1:],
                            input_spec.get_network_input_type(),
                            input_spec.get_id())


def reduce_tensor_batch_dimension(tensor: Tensor) -> tuple:
    """
    Helper function to reduce tensor batch dimension.
    """
    if len(tensor.shape) > 1:
        return tensor.shape[1:]
    else:
        return tensor.shape
