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


from abc import ABCMeta, abstractmethod
from kat_api.observation import IObservation
from kat_typing import Action
from typing import Collection, Tuple
from kat_api.prop_desc import ITensorDescriptor


class IGame(metaclass=ABCMeta):
    """
    Common interface for game wrappers.
    Framework is assuming episode based games, with reward per action step taken.
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
        return (hasattr(subclass, 'init') and
                callable(subclass.init) and
                hasattr(subclass, 'reset') and
                callable(subclass.reset) and
                hasattr(subclass, 'make_action') and
                callable(subclass.make_action) and
                hasattr(subclass, 'get_current_state') and
                callable(subclass.get_current_state) and
                hasattr(subclass, 'is_initialized') and
                callable(subclass.is_initialized) and
                hasattr(subclass, 'is_episode_finished') and
                callable(subclass.is_episode_finished) and
                hasattr(subclass, 'get_action_space_desc') and
                callable(subclass.get_action_space_desc) and
                hasattr(subclass, 'get_observation_space_desc') and
                callable(subclass.get_observation_space_desc) and
                hasattr(subclass, 'process_ticks') and
                callable(subclass.process_ticks) and
                hasattr(subclass, 'get_total_score') and
                callable(subclass.get_total_score) or
                NotImplemented)

    @abstractmethod
    def init(self) -> None:
        """
        Resets the game, and returns the initial state.

        :return:
            the initial state (IState)
        """
        pass

    @abstractmethod
    def reset(self) -> IObservation:
        """
        Resets the game, and returns the initial state.

        :return:
            the initial observation state (IObservation)
        """
        pass

    @abstractmethod
    def make_action(self, action: Action) -> Tuple[IObservation, float]:
        """
        Apply an action on the environment and return the new state.

        :param action:
            the action which should be applied
        :return:
            the next observation state
        """
        pass

    @abstractmethod
    def get_current_observation(self) -> IObservation:
        """
        Returns the actual observation object.

        :return:
         the current state
        """
        pass

    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Object state indicator.

        :returns
            True if the "init" method was called, otherwise false.
        """
        pass

    @abstractmethod
    def is_episode_finished(self) -> bool:
        """
        Environment state indicator.

        :returns
            True if the game's current episode is finished, otherwise false.
        """
        pass

    @abstractmethod
    def get_action_space_desc(self) -> ITensorDescriptor:
        """
        Retrieves the available actions space descriptor for the current game.

        :returns
            the actions space descriptor
        """
        pass

    @abstractmethod
    def get_observation_space_desc(self) -> Collection[ITensorDescriptor]:
        """
        Retrieves the available observation space descriptors for the current game.
        Assuming that environments can have multiple observation outputs. For example:
        frame_buffer, z_buffer, ram_footprint, etc...
        :returns
            collection of observation space descriptors
        """
        pass

    @abstractmethod
    def process_ticks(self, num_of_ticks: int) -> None:
        """
        Processes a specified number of tics with the last action chosen by the agent.
        """
        pass

    @abstractmethod
    def get_total_score(self) -> float:
        """
        Retrieves the total score of an episode.

        :returns
            the total score in floating point format
        """
        pass
