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
from enum import Enum
from kat_api.observation import IObservation
from kat_typing import Action


class StateType(Enum):
    """
    Abstract base enum for state types.
    """
    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    """
    Basic possible game states:
    * NONE_STATE :
        undefined state
    * INITIAL_STATE :
        game.init() state
    * ACTIVE_STATE :
        any state after the initial state
    * END_STATE :
        last state which can be observed
    """
    NONE_STATE = "none_state"
    INITIAL_STATE = "initial_state"
    ACTIVE_STATE = "active_state"
    END_STATE = "end_state"


class IState(metaclass=ABCMeta):
    """
    Common interface for states.

    An `IState` contains the data emitted by a game environment at each action.

    An `IState` possibly holds a `StateType`, an `IObservation`,
    and an associated `reward` and `discount`.
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
        return (
                hasattr(subclass, 'get_state_id') and
                callable(subclass.get_state_id) and
                hasattr(subclass, 'bump_state_id') and
                callable(subclass.bump_state_id) and
                hasattr(subclass, 'get_transitioned_observation') and
                callable(subclass.get_transitioned_observation) and
                hasattr(subclass, 'set_transitioned_observation') and
                callable(subclass.set_transitioned_observation) and
                hasattr(subclass, 'get_reward') and
                callable(subclass.get_reward) and
                hasattr(subclass, 'set_reward') and
                callable(subclass.set_reward) and
                hasattr(subclass, 'get_transition') and
                callable(subclass.get_transition) and
                hasattr(subclass, 'set_transition') and
                callable(subclass.set_transition) and
                hasattr(subclass, 'get_discount') and
                callable(subclass.get_discount) and
                hasattr(subclass, 'get_state_type') and
                callable(subclass.get_state_type) and
                hasattr(subclass, 'is_initial_state') and
                callable(subclass.is_initial_state) and
                hasattr(subclass, 'is_end_state') and
                callable(subclass.is_end_state) or
                NotImplemented)

    @abstractmethod
    def get_state_id(self) -> int:
        """
        :returns:
            unique identifier
        """
        pass

    @abstractmethod
    def get_observation(self) -> IObservation:
        """
        :returns:
            current observation of the environment
        """
        pass

    @abstractmethod
    def get_transitioned_observation(self) -> IObservation:
        """
        :returns:
            transitioned observation (after "take_action")
        """
        pass

    @abstractmethod
    def set_transitioned_observation(self, transitioned_observation: IObservation) -> None:
        """
        Sets the transitioned observation (after "take_action")
        """
        pass

    @abstractmethod
    def get_reward(self) -> float:
        """
        :return:
            reward for the action taken
        """
        pass

    @abstractmethod
    def set_reward(self, reward: float) -> None:
        """
        Sets the reward
        """
        pass

    @abstractmethod
    def get_transition(self) -> Action:
        """
        :returns:
            the selected action between observation and transitioned observation
        """
        pass

    @abstractmethod
    def set_transition(self, action: Action) -> None:
        """
        Sets the selected action.
        """
        pass

    @abstractmethod
    def get_discount(self) -> float:
        pass

    @abstractmethod
    def get_state_type(self) -> StateType:
        """
        :returns
            State indicator. (initial, active, end)
        """
        pass

    @abstractmethod
    def is_initial_state(self) -> bool:
        """
        :returns:
             true if the state is an initial state, else false
        """
        pass

    @abstractmethod
    def is_end_state(self) -> bool:
        """
        :returns:
             true if the state is an end state, else false
        """
        pass
