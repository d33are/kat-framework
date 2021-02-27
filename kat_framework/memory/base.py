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

from kat_api import ITensorDescriptor
from kat_typing import Tensor
from abc import ABCMeta, abstractmethod
from typing import Tuple
from kat_framework.util import tensors
import numpy as np


class BaseMemory(metaclass=ABCMeta):
    """
    Abstract base implementation for `IReplayMemory`.
    """

    # protected members

    _num_of_buffers: int = 0
    _S1_BUFFER_NAME: str = "s1_states"
    _A_BUFFER_NAME: str = "action_ids"
    _S2_BUFFER_NAME: str = "s2_states"
    _R_BUFFER_NAME: str = "rewards"
    _T_BUFFER_NAME: str = "terminals"
    _s1_states: np.ndarray = None
    _s1_states_spec: ITensorDescriptor = None
    _action_ids: np.ndarray = None
    _action_ids_spec: ITensorDescriptor = None
    _s2_states: np.ndarray = None
    _s2_states_spec: ITensorDescriptor = None
    _rewards: np.ndarray = None
    _rewards_spec: ITensorDescriptor = None
    _terminals: np.ndarray = None
    _terminals_spec: ITensorDescriptor = None
    _deep: int = 0
    _max_capacity: int = 0

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        self._num_of_buffers = 5

    def init(self, buffer_spec: Tuple[ITensorDescriptor]) -> None:
        """
        Object initialization.

        # see : IReplayMemory.init(buffer_spec)

        :param buffer_spec:
            layout specifications of the buffers
        """
        if buffer_spec is None:
            raise ValueError("No buffer_spec specified.")
        if len(buffer_spec) < self._num_of_buffers:
            raise ValueError("At least {} buffer(s) must be specified.".format(self._num_of_buffers))
        for descriptor in buffer_spec:
            if self._S1_BUFFER_NAME == descriptor.get_display_name():
                self._s1_states = np.empty(descriptor.get_tensor_shape(), descriptor.get_data_type())
                self._s1_states_spec = tensors.reduce_spec_batch_dimension(descriptor)
            if self._A_BUFFER_NAME == descriptor.get_display_name():
                self._action_ids = np.empty(descriptor.get_tensor_shape(), descriptor.get_data_type())
                self._action_ids_spec = tensors.reduce_spec_batch_dimension(descriptor)
            if self._S2_BUFFER_NAME == descriptor.get_display_name():
                self._s2_states = np.empty(descriptor.get_tensor_shape(), descriptor.get_data_type())
                self._s2_states_spec = tensors.reduce_spec_batch_dimension(descriptor)
            if self._R_BUFFER_NAME == descriptor.get_display_name():
                self._rewards = np.empty(descriptor.get_tensor_shape(), descriptor.get_data_type())
                self._rewards_spec = tensors.reduce_spec_batch_dimension(descriptor)
            if self._T_BUFFER_NAME == descriptor.get_display_name():
                self._terminals = np.empty(descriptor.get_tensor_shape(), descriptor.get_data_type())
                self._terminals_spec = tensors.reduce_spec_batch_dimension(descriptor)
        self._max_capacity = len(self._s1_states)

    def get_number_of_frames(self) -> int:
        """
        Total number of frames stored in the buffer.

        # see : IReplayMemory.get_number_of_frames()
        """
        return self._deep

    def get_all(self) -> Tuple[np.ndarray,
                               np.ndarray,
                               np.ndarray,
                               np.ndarray,
                               np.ndarray]:
        """
        Gets all items, from all buffers.

        # see : IReplayMemory.get_all()
        """
        s1_samples = self._s1_states.copy()
        a_samples = self._action_ids.copy()
        s2_samples = self._s2_states.copy()
        r_samples = self._rewards.copy()
        t_samples = self._terminals.copy()
        return s1_samples, a_samples, s2_samples, r_samples, t_samples

    def add_transition(self,
                       s1_state: Tensor,
                       action_idx: int,
                       s2_state: Tensor,
                       reward: float,
                       is_end_state: bool) -> None:
        """
        Adds a transition to the specified buffers.

        # see : IReplayMemory.add_transition()
        """
        if not tensors.check_same_tensor_structure(s1_state, self._s1_states_spec):
            raise ValueError("s1 state input vs s1 state spec mismatch")
        if not tensors.check_same_tensor_structure(s2_state, self._s2_states_spec):
            raise ValueError("s2 state input vs s2 state spec mismatch")
        return self._add_transition(s1_state, action_idx, s2_state, reward, is_end_state)

    # protected member functions

    @abstractmethod
    def _add_transition(self,
                        s1_state: Tensor,
                        action_idx: int,
                        s2_state: Tensor,
                        reward: float,
                        is_end_state: bool) -> None:
        """
        Abstract method for derived classes.

        :param s1_state:
            current initiator state
        :param action_idx:
            current action index
        :param s2_state:
            current transitioned state
        :param reward:
            current reward
        :param is_end_state:
            is end state or not
        """
        pass
