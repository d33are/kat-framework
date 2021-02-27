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

from kat_api import IReadOnlyMemory, IReplayMemory
from kat_typing import IterableDataset
from kat_framework.util import logger


log = logger.get_logger(__name__ + ".MemoryAccessor")


class MemoryAccessor(IReadOnlyMemory):
    """
    Read only replay memory wrapper.
    """

    # protected members

    _replay_memory: IReplayMemory = None

    # public members

    def __init__(self):
        """
        Default constructor.
        """
        pass

    def init(self, replay_memory: IReplayMemory) -> None:
        """
        Object initialization.

        :param replay_memory:
            replay memory instance
        """
        self._replay_memory = replay_memory

    def as_iterable_dataset(self, input_context: object = None) -> IterableDataset:
        """
        Get sample as iterable dataset.

        :param input_context:
            input context if specified (for ex. tensorflow)
        :return:
            iterable dataset instance
        """
        return self._replay_memory.as_iterable_dataset(input_context)
