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

from kat_framework.framework import KatherineApplication
from kat_framework.util import logger
from kat_api import IObservation, ITensorDescriptor, IConfigurationHandler
from kat_typing import Action, Tensor
from abc import ABCMeta, abstractmethod
from typing import List, Tuple
from logging import Logger


class Game(metaclass=ABCMeta):
    """
    Abstract base class for game implementations.
    """

    # protected members

    _log: Logger = None
    _config_handler: IConfigurationHandler = None
    _game_instance = None  # object
    _initialized: bool = False
    _current_observation: IObservation = None
    _action_space: Tensor = None
    _action_space_desc: ITensorDescriptor = None
    _observation_space_desc: List[ITensorDescriptor] = None

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        self._config_handler = KatherineApplication.get_application_config()
        if not self._config_handler:
            raise ValueError("No config_handler specified")
        self._log = logger.get_logger(self.__class__.__name__)

    def init(self) -> None:
        """
        Object initialization.
        """
        self._init_config()

    def make_action(self, action: Action) -> Tuple[IObservation, float]:
        """
        # see: IGame.make_action()
        """
        if action is None:
            raise ValueError("No action specified")
        return self._make_action(action)

    def get_current_observation(self) -> IObservation:
        """
        # see: IGame.get_current_observation()
        """
        return self._current_observation

    def is_initialized(self) -> bool:
        """
        # see: IGame.is_initialized()
        """
        return self._initialized

    def get_action_space_desc(self) -> ITensorDescriptor:
        """
        # see: IGame.get_action_space_desc()
        """
        self._init_check()
        if self._action_space_desc is None:
            self._action_space_desc = self._build_actions_space_desc()
        return self._action_space_desc

    # protected member functions

    @abstractmethod
    def _init_config(self) -> None:
        """
        Loads necessary configurations, for the game.
        """
        pass

    @abstractmethod
    def _build_actions_space_desc(self) -> ITensorDescriptor:
        """
        Builds action space descriptor based on the game's action space.

        :returns:
           action space descriptor
        """
        pass

    @abstractmethod
    def _make_action(self, action: Action) -> Tuple[IObservation, float]:
        """
        Takes 1 step in the environment with the provided action.

        :param action:
            chosen action
        :return:
            next observation state, reward for the action
        """
        pass

    def _init_check(self) -> None:
        """
        Checks that the game is initialised or not.

        :raises RuntimeError
            if the game instance was not initialized (self._initialized == False)
        """
        if not self._initialized:
            raise RuntimeError("Game is not initialized.")
