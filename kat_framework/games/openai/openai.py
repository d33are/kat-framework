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

from kat_framework.config.config_props import OpenAIConfigurationProperty
from kat_framework.core.descriptors import TensorDescriptor
from kat_framework.games.base import Game
from kat_framework.games.openai.observation import OpenAIObservation
from kat_api import IGame, ITensorDescriptor, IObservation, NetworkInputType
from kat_typing import Action
from typing import Collection
from overrides import overrides
from typing import Tuple
from enum import Enum
import numpy as np
import gym

SCREEN_BUFFER = "screen_buffer"
RAM_VECTOR = "ram_vector"


class OpenAIGame(Enum):
    """
    OpenAI game descriptors.

    Properties:
     * env name
     * network input (image or ram)
     * buffer name
    """

    def __new__(cls, label, network_type, buffer_name):
        obj = object.__new__(cls)
        obj.label = label
        obj.network_type = network_type
        obj.buffer_name = buffer_name
        obj._value_ = label
        return obj

    LUNAR_LANDER_V2 = ("LunarLander-v2", NetworkInputType.RAM, RAM_VECTOR)
    BREAKOUT_V0 = ("Breakout-v0", NetworkInputType.IMG, SCREEN_BUFFER)
    BREAKOUT_RAM_V0 = ("Breakout-ram-v0", NetworkInputType.RAM, RAM_VECTOR)
    SPACE_INVADERS_V0 = ("SpaceInvaders-v0", NetworkInputType.IMG, SCREEN_BUFFER)
    SPACE_INVADERS_RAM_V0 = ("SpaceInvaders-ram-v0", NetworkInputType.RAM, RAM_VECTOR)
    PONG_V0 = ("Pong-v0", NetworkInputType.IMG, SCREEN_BUFFER)
    PONG_RAM_V0 = ("Pong-ram-v0", NetworkInputType.RAM, RAM_VECTOR)
    ATLANTIS_V0 = ("Atlantis-v0", NetworkInputType.IMG, SCREEN_BUFFER)
    ATLANTIS_RAM_V0 = ("Atlantis-ram-v0", NetworkInputType.RAM, RAM_VECTOR)
    ENDURO_V0 = ("Enduro-v0", NetworkInputType.IMG, SCREEN_BUFFER)
    ENDURO_RAM_V0 = ("Enduro-ram-v0", NetworkInputType.RAM, RAM_VECTOR)
    FREEWAY_V0 = ("Freeway-v0", NetworkInputType.IMG, SCREEN_BUFFER)
    FREEWAY_RAM_V0 = ("Freeway-ram-v0", NetworkInputType.RAM, RAM_VECTOR)
    MS_PACMAN_V0 = ("MsPacman-v0", NetworkInputType.IMG, SCREEN_BUFFER)
    MS_PACMAN_RAM_V0 = ("MsPacman-ram-v0", NetworkInputType.RAM, RAM_VECTOR)
    PHOENIX_V0 = ("Phoenix-v0", NetworkInputType.IMG, SCREEN_BUFFER)
    PHOENIX_RAM_V0 = ("Phoenix-ram-v0", NetworkInputType.RAM, RAM_VECTOR)
    SOLARIS_V0 = ("Solaris-v0", NetworkInputType.IMG, SCREEN_BUFFER)
    SOLARIS_RAM_V0 = ("Solaris-ram-v0", NetworkInputType.RAM, RAM_VECTOR)


class OpenAI(Game, IGame):
    """
    OpenAI gym game wrapper.
    """

    # protected members

    _last_action: Action = None
    _total_reward: float = 0.
    _done: bool = False
    _env_name: str = None
    _render_enabled: bool = False
    _game_spec: OpenAIGame = None
    _actions_space_id: str = "action_space"

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        super(OpenAI, self).__init__()

    @staticmethod
    def _check_env_is_supported(env_name) -> None:
        """
        Checks that the given environment can be found in the descriptor "list".
        :param env_name:
            specified env name
        :raises RuntimeError
            when the specified env name can't be found in the descriptor list
        """
        if OpenAIGame(env_name) is None:
            raise RuntimeError("Environment {} is not supported".format(env_name))

    @overrides
    def init(self) -> None:
        """
        Object initialization.
        """
        super(OpenAI, self).init()
        self._initialized = True

    @overrides
    def process_ticks(self, num_of_ticks: int) -> None:
        """
        # see : IGame.process_ticks(num_of_ticks)
        """
        if num_of_ticks > 0:
            [self.make_action(self.last_action) for i in range(num_of_ticks)]

    @overrides
    def reset(self) -> IObservation:
        """
        # see : IGame.reset()
        """
        initial_state = self._game_instance.reset()
        self._current_observation = OpenAIObservation(initial_state, self._game_spec.network_type)
        self._total_reward = 0
        return self._current_observation

    @overrides
    def is_episode_finished(self) -> bool:
        """
        # see : IGame.is_episode_finished()
        """
        return self.done

    @overrides
    def get_observation_space_desc(self) -> Collection[ITensorDescriptor]:
        """
        # see : IGame.get_observation_space_desc()
        """
        self._init_check()
        if self._observation_space_desc is None:
            self._observation_space_desc = []
            self._observation_space_desc.append(
                TensorDescriptor(display_name=self._game_spec.buffer_name,
                                 data_type=self._game_instance.observation_space.dtype,
                                 tensor_shape=self._game_instance.observation_space.shape,
                                 input_type=self._game_spec.network_type))
        return self._observation_space_desc

    @overrides
    def get_total_score(self):
        """
        # see : IGame.get_total_score()
        """
        return self._total_reward

    # protected members

    @overrides
    def _init_config(self) -> None:
        """
        Loads the configuration.
        """
        self._env_name = self._config_handler.get_config_property(OpenAIConfigurationProperty.ENV_NAME,
                                                                  OpenAIConfigurationProperty.ENV_NAME.prop_type)
        self._game_spec = OpenAIGame(self._env_name)
        self._render_enabled = \
            self._config_handler.get_config_property(OpenAIConfigurationProperty.RENDER_ENABLED,
                                                     OpenAIConfigurationProperty.RENDER_ENABLED.prop_type)
        self._check_env_is_supported(self._env_name)
        self._game_instance = gym.make(self._env_name)

    @overrides
    def _make_action(self, action: Action) -> Tuple[IObservation, float]:
        """
        # see Game._make_action(action)

        :param action:
             specified action to be executed on the environment
        :returns:
            next observation, reward value
        """
        observation, reward, self.done, _ = self._game_instance.step(action)
        self._total_reward += reward
        if observation is not None:
            self._current_observation = OpenAIObservation(observation, self._game_spec.network_type)
        else:
            self._current_observation = None
        self.last_action = action
        if self._render_enabled:
            self._game_instance.render()
        return self._current_observation, reward

    @overrides
    def _build_actions_space_desc(self):
        """
        # see Game._build_actions_space_desc()

        :returns:
            action space tensor descriptor
        :raises NotImplemented
            if the provided action space form a gym environment is not box or discrete
            action space
        """
        if isinstance(self._game_instance.action_space, gym.spaces.Box):
            return TensorDescriptor(display_name=self._actions_space_id,
                                    data_type=self._game_instance.action_space.dtype,
                                    tensor_shape=self._game_instance.action_space.shape,
                                    input_type=NetworkInputType.NONE,
                                    low=self._game_instance.action_space.low,
                                    high=self._game_instance.action_space.high)
        elif isinstance(self._game_instance.action_space, gym.spaces.Discrete):
            return TensorDescriptor(display_name=self._actions_space_id,
                                    data_type=np.int32,
                                    tensor_shape=(self._game_instance.action_space.n,),
                                    input_type=NetworkInputType.NONE)
        else:
            raise NotImplemented
