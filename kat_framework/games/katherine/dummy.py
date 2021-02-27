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

from kat_framework.core.descriptors import TensorDescriptor
from kat_framework.games.base import Game
from kat_framework.games.katherine.observation import DummyObservation
from kat_api import IGame, ITensorDescriptor, IObservation, NetworkInputType
from kat_typing import Action
from typing import Collection
from overrides import overrides
from typing import Tuple
import random
import numpy as np

SCREEN_BUFFER = "screen_buffer"
SCREEN_WEIGHT = 640
SCREEN_HEIGHT = 480
SCREEN_CHANNELS = 3
MIN_REWARD = 0.0
MAX_REWARD = 100.0
ACTION_SPACE_DIMENSION = 4


class DummyGame(Game, IGame):
    """
    Dummy game wrapper.

    Mainly for testing purposes.
    """

    # protected members

    _done: bool = None
    _total_reward: float = 0.0

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        super(DummyGame, self).__init__()

    @overrides
    def init(self) -> None:
        """
        # see : IGame.init()
        """
        super(DummyGame, self).init()
        self._initialized = True

    @overrides
    def process_ticks(self, num_of_ticks: int) -> None:
        """
        # see : IGame.process_ticks(num_of_ticks)
        """
        pass

    @overrides
    def reset(self) -> IObservation:
        """
        # see : IGame.process_ticks(num_of_ticks)
        """
        self._total_reward = 0.0
        return DummyObservation(np.random.rand(SCREEN_WEIGHT, SCREEN_HEIGHT, SCREEN_CHANNELS).astype(np.float32))

    @overrides
    def is_episode_finished(self) -> bool:
        """
        # see : IGame.is_episode_finished()
        """
        return self._done

    @overrides
    def get_observation_space_desc(self) -> Collection[ITensorDescriptor]:
        """
        # see : IGame.get_observation_space_desc()
        """
        self._init_check()
        if self._observation_space_desc is None:
            self._observation_space_desc = []
            self._observation_space_desc.append(
                TensorDescriptor(display_name="screen_buffer",
                                 data_type=np.float32,
                                 tensor_shape=(SCREEN_WEIGHT, SCREEN_HEIGHT, SCREEN_CHANNELS),
                                 input_type=NetworkInputType.IMG))
        return self._observation_space_desc

    @overrides
    def get_total_score(self):
        """
        # see : IGame.get_total_score()
        """
        return self._total_reward

    # protected member functions

    @overrides
    def _make_action(self, action: Action) -> Tuple[IObservation, float]:
        """
        # see : Game._make_action(action)
        """
        # TODO: config for randomized values
        observation = np.random.rand(SCREEN_WEIGHT, SCREEN_HEIGHT, SCREEN_CHANNELS).astype(np.float32)
        reward = np.random.uniform(MIN_REWARD, MAX_REWARD)
        self._done = random.choice([True, False])
        self._total_reward += reward
        self._current_observation = DummyObservation(observation)
        return self._current_observation, reward

    @overrides
    def _build_actions_space_desc(self):
        """
        # see : Game._build_actions_space_desc()
        """
        return TensorDescriptor(display_name="action_space",
                                data_type=np.int32,
                                tensor_shape=(ACTION_SPACE_DIMENSION,),
                                input_type=NetworkInputType.NONE)

    @overrides
    def _init_config(self) -> None:
        """
        # see : Game._init_config()
        """
        pass
