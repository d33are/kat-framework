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

from kat_framework.memory.base import BaseMemory
from kat_framework.util import logger
from kat_api import ITensorDescriptor, IReplayMemory
from kat_typing import Tensor, IterableDataset
from typing import Tuple, List
from overrides import overrides
from logging import Logger
import numpy as np
import uuid


class UniformMemory(BaseMemory, IReplayMemory):
    """
    Uniform replay memory implementation.

    This implementation is based on circular buffer pattern.
    """

    # protected members

    _log: Logger = None
    _transition_ids: List = None

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        super(UniformMemory, self).__init__()
        self._log = logger.get_logger(self.__class__.__name__)

    def init(self, buffer_spec: Tuple[ITensorDescriptor]):
        """
        Object initialization.

        # see : IReplayMemory.init(buffer_spec)
        """
        super(UniformMemory, self).init(buffer_spec)
        self._transition_ids = [None] * self._max_capacity

    @overrides
    def as_iterable_dataset(self, input_context: object = None) -> IterableDataset:
        """
        # see : IReplayMemory.as_iterable_dataset(input_context)
        """
        raise NotImplemented

    @overrides
    def reset(self) -> None:
        """
        # see : IReplayMemory.reset()
        """
        raise NotImplemented

    @overrides
    def get_sample(self, sample_size: int) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
        """
        # see : IReplayMemory.get_sample(sample_size)
        """
        max_batch_size = min(self._deep, self._max_capacity)
        if max_batch_size == 0:
            max_batch_size = sample_size
        batch_index = np.random.choice(max_batch_size, sample_size, replace=False)
        s1_samples = self._s1_states[batch_index]
        a_samples = self._action_ids[batch_index]
        s2_samples = self._s2_states[batch_index]
        r_samples = self._rewards[batch_index]
        t_samples = self._terminals[batch_index]
        return s1_samples, a_samples, s2_samples, r_samples, t_samples

    # protected member functions

    @overrides
    def _add_transition(self,
                        s1_state: Tensor,
                        action_idx: int,
                        s2_state: Tensor,
                        reward: float,
                        is_end_state: bool) -> None:
        """
        # see : BaseMemory._add_transition()
        """
        transition_id = str(uuid.uuid1())
        circular_index = self._deep % self._max_capacity
        self._transition_ids[circular_index] = transition_id
        self._s1_states[circular_index] = s1_state
        self._action_ids[circular_index] = action_idx
        self._s2_states[circular_index] = s2_state
        self._rewards[circular_index] = reward
        self._terminals[circular_index] = is_end_state
        self._deep += 1
