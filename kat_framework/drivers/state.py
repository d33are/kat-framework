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
from kat_typing import Action
from kat_api import IObservation, IState, StateType
from overrides import overrides


class KatState(IState):
    """
    Default `IState` implementation.

    An `IState` contains a time step of the full episode trajectory.
    """

    # protected members

    _observation: IObservation = None
    _transitioned_observation: IObservation = None
    _state_type: StateType = None
    _transition: Action = None
    _reward: float = float(-1.0)
    _discount: float = float(0.99)
    _state_id: int = 0

    # public member functions

    def __init__(self, state_id: int, observation: IObservation, state_type: StateType, discount: float = None):
        """
        Default constructor.

        :param state_id:
            unique state id
        :param observation:
            initiator observation
        :param state_type:
            state type
        :param discount:
            discount factor
        """
        self._observation = observation
        self._state_type = state_type
        self._discount = discount
        self._state_id = state_id

    def __eq__(self, other):
        if isinstance(other, KatState):
            return self.get_state_id() == other.get_state_id()
        return False

    @overrides
    def get_state_id(self) -> int:
        """
        # see IState.get_state_id()
        """
        return self._state_id

    @overrides
    def get_observation(self) -> IObservation:
        """
        # see IState.get_observation()
        """
        return self._observation

    @overrides
    def get_transitioned_observation(self) -> IObservation:
        """
        # see IState.get_transitioned_observation()
        """
        return self._transitioned_observation

    @overrides
    def set_transitioned_observation(self, transitioned_observation: IObservation) -> None:
        """
        # see IState.set_transitioned_observation()
        """
        self._transitioned_observation = transitioned_observation

    @overrides
    def get_reward(self) -> float:
        """
        # see IState.get_reward()
        """
        return self._reward

    @overrides
    def set_reward(self, reward: float) -> None:
        """
        # see IState.set_reward()
        """
        self._reward = reward

    @overrides
    def get_transition(self) -> Action:
        """
        # see IState.get_transition()
        """
        return self._transition

    @overrides
    def set_transition(self, action: Action) -> None:
        """
        # see IState.set_transition()
        """
        self._transition = action

    @overrides
    def get_discount(self) -> float:
        """
        # see IState.get_discount()
        """
        return self._discount

    @overrides
    def get_state_type(self) -> StateType:
        """
        # see IState.get_state_type()
        """
        return self.state_type

    @overrides
    def is_initial_state(self) -> bool:
        """
        # see IState.is_initial_state()
        """
        return self.state_type == StateType.INITIAL_STATE

    @overrides
    def is_end_state(self) -> bool:
        """
        # see IState.is_end_state()
        """
        return self.state_type == StateType.END_STATE

    @property
    def state_type(self):
        """
        State type accessor.
        """
        return self._state_type

    @state_type.setter
    def state_type(self, value):
        """
        State mutator for state type objects.

        Controls the state transition workflow as follows:

        * `NONE_STATE`:
            from any state
        * `INITIAL_STATE`:
            from NONE_STATE
        * `ACTIVATE_STATE`:
            from INITIAL_STATE
        * `END_STATE`:
            from ACTIVATE_STATE

        If the state transition violates these rules, the new value won't be accepted. (so nothing will happen)
        """
        if (value == StateType.INITIAL_STATE and (self._state_type != StateType.NONE_STATE or (
                self.state_type is not None))) or (
                value == StateType.ACTIVE_STATE and self._state_type != StateType.INITIAL_STATE) or (
                value == StateType.END_STATE and self._state_type != StateType.ACTIVE_STATE
        ):
            return
        self._state_type = value or StateType.NONE_STATE

    def __str__(self):
        output = ""
        for _, var in vars(self).items():
            output += str(var) + "::"
        return output

