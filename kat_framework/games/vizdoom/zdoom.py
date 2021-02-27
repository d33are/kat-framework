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
from typing import Collection

from kat_framework.games.base import Game
from kat_framework.config.config_props import ViZDoomConfigurationProperty
from kat_framework.core.descriptors import TensorDescriptor
from kat_framework.games.vizdoom.observation import DoomObservation
from kat_api import IGame, ITensorDescriptor, IObservation, NetworkInputType
from kat_typing import Action
from vizdoom.vizdoom import DoomGame, AMMO2, HEALTH, KILLCOUNT
from overrides import overrides
from typing import Tuple
import numpy as np

CONFIGURATION_ATTRIBUTE_ERROR_MSG = "ViZDoom has no attribute named: %s"
SET_ATTRIBUTE_PREFIX = "set_"


class ViZDoomGame(Game, IGame):
    """
    VizDoom game wrapper.
    """

    # protected members

    _action_space_id: str = "action_space"
    _last_ammo_2: int = 0
    _last_kill_count: int = 0
    _last_health: int = 0

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        super(ViZDoomGame, self).__init__()
        self._game_instance = DoomGame()

    @overrides
    def init(self) -> None:
        """
        Object initialization.
        """
        super(ViZDoomGame, self).init()
        self._game_instance.init()
        self._initialized = True

    @overrides
    def reset(self) -> IObservation:
        """
        # see : IGame.reset()
        """
        self._game_instance.new_episode()
        initial_state = self._game_instance.get_state()
        self._current_observation = DoomObservation(initial_state)
        return self._current_observation

    @overrides
    def process_ticks(self, num_of_ticks: int) -> None:
        """
        # see : IGame.process_ticks()
        """
        self._game_instance.advance_action(num_of_ticks, True)
        self._update_current_observation()

    @overrides
    def is_episode_finished(self) -> bool:
        """
        # see : IGame.is_episode_finished()
        """
        return self._game_instance.is_episode_finished()

    @overrides
    def get_observation_space_desc(self) -> Collection[ITensorDescriptor]:
        """
        # see : IGame.get_observation_space_desc()

        Currently we are filtering out non array like buffers.
        """
        self._init_check()
        if self._observation_space_desc is None:
            self._observation_space_desc = []
            dummy_state = self._game_instance.get_state()
            observed_attributes = \
                filter(lambda a:
                       not a.startswith('__')
                       and not a.startswith('tic')
                       and not a.startswith('number')
                       and not a.startswith('labels')
                       and not a.startswith('objects')
                       and not a.startswith('sectors')
                       and not callable(getattr(dummy_state, a)), dir(dummy_state))
            for attr in observed_attributes:
                tensor = np.array(getattr(dummy_state, attr))
                self._observation_space_desc.append(TensorDescriptor(display_name=attr,
                                                                     data_type=tensor.dtype,
                                                                     tensor_shape=tensor.shape,
                                                                     input_type=NetworkInputType.IMG))
        return self._observation_space_desc

    @overrides
    def get_total_score(self):
        """
        # see : IGame.get_total_score()
        """
        return self._game_instance.get_total_reward()

    # protected member functions

    @overrides
    def _init_config(self) -> None:
        """
        Initializes the game instance, based on the config_handler attributes.
        """
        for config_property in ViZDoomConfigurationProperty:
            config_value = self._config_handler.get_config_property(config_property, config_property.prop_type)
            if config_value is None:
                continue
            try:
                getattr(self._game_instance, SET_ATTRIBUTE_PREFIX + config_property.label)(config_value)
            except (AttributeError, TypeError):
                self._log.warning(CONFIGURATION_ATTRIBUTE_ERROR_MSG, config_property.label)

    @overrides
    def _make_action(self, action: Action) -> Tuple[IObservation, float]:
        """
        # see : Game._make_action()
        """
        reward = self._game_instance.make_action(action)
        self._update_current_observation()
        self._update_variables()
        return self._current_observation, reward

    @overrides
    def _build_actions_space_desc(self):
        """
        # see : Game._build_actions_space_desc()
        """
        return TensorDescriptor(display_name=self._action_space_id,
                                data_type=np.int32,
                                tensor_shape=(self._game_instance.get_available_buttons_size(),),
                                input_type=NetworkInputType.NONE)

    def _update_current_observation(self) -> None:
        """
        Updates the current observation based on the environment.
        """
        observation = self._game_instance.get_state()
        if observation is not None:
            self._current_observation = DoomObservation(observation)

    def _update_variables(self) -> None:
        """
        Updates game variables.
        """
        self._last_ammo_2 = self._game_instance.get_game_variable(AMMO2)
        self._last_health = self._game_instance.get_game_variable(HEALTH)
        self._last_kill_count = self._game_instance.get_game_variable(KILLCOUNT)
