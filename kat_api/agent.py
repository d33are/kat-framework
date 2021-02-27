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
from kat_api.state import IState
from kat_typing import Action, TrainLoss, DistributionStrategy
from typing import Collection
from kat_api.prop_desc import ITensorDescriptor


class IAgent(metaclass=ABCMeta):
    """
    Core logics are implemented as agents. An agent can "observe" game states,
    based on that experience they can define/learn/train a policy to interact with the
    given environment(s).
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
                hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'take_action') and
                callable(subclass.take_action) and
                hasattr(subclass, 'is_initialized') and
                callable(subclass.is_initialized) and
                hasattr(subclass, 'store_transition') and
                callable(subclass.store_transition) and
                hasattr(subclass, 'get_distribution_strategy') and
                callable(subclass.get_distribution_strategy) and
                hasattr(subclass, 'persist_model') and
                callable(subclass.persist_model) and
                hasattr(subclass, 'get_exploration_rate') and
                callable(subclass.get_exploration_rate) or
                NotImplemented)

    @abstractmethod
    def init(self,
             observation_space_desc: Collection[ITensorDescriptor],
             action_space_desc: ITensorDescriptor) -> None:
        """
        We're assuming that Agents gonna' be instantiated through factory interfaces. Based on that
        knowledge, an ideal Agent implementation has a "no args" constructor, and a separated initialization
        method. (object instantiation is _NOT_ initialization)

        :param observation_space_desc
            Collection of observation (mostly tensors) descriptors, in most of the cases these are provided
            by the given environment.

        :param action_space_desc
            Action space descriptor, currently assuming discrete action space.
        """
        pass

    @abstractmethod
    def tick(self, current_episode: int, current_step: int) -> None:
        """
        Assuming that the train function is able to execute as a separate thread, so the agent provides an
        interface for updating the necessary internal states.

        :param current_episode:
            current episode number
        :param current_step:
            current step number in the environment
        """

    @abstractmethod
    def train(self) -> TrainLoss:
        """
        Trains the agent's policy for optimal environment interactions. This Agent interface is designed for
        "episodic step" drivers, so we're assuming to do one train step per call.
        Implementations needed to be prepared for both synchronous and asynchronous execution.

        :returns
            Current loss value after executing the train step, provided by the loss function.
        """
        pass

    @abstractmethod
    def take_action(self, game_state: IState) -> Action:
        """
        This Agent interface is designed for "episodic step" drivers, so we're assuming to do one time step per call.
        Based on the given input state, the Agent must evaluate with policies or networks, and take an action.

        :returns
            action taken by the agent for the current time step (MDP)
        """
        pass

    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Agent's initialization state indicator.

        :returns
            True if the "init" method was called, otherwise false.
        """
        pass

    @abstractmethod
    def store_transition(self, state: IState) -> str:
        """
        We're assuming that Drivers can "tell" the Agent to store/process/ignore the current game state.
        For example experience replay based agents can store the current game state to the associated
        replay memory.

        :param state
            Current game state provided by the Driver.

        :returns
            the unique ID of the stored experience (acts like a primary key)
        """
        pass

    @abstractmethod
    def get_distribution_strategy(self) -> DistributionStrategy:
        """
        Modern ML APIs can distribute training across multiple GPUs, multiple machines. It means
        the implementation must able to "know" that it is training on a single machine or on a cluster.

        :returns:
            the agent's distribution strategy
        """
        pass

    @abstractmethod
    def persist_model(self) -> None:
        """
        Assuming, the Agent is able to persist its policy or network in a stateful manner.
        This can be:
            * saving policy tables
            * saving model weights
            * saving whole model graphs
            ... and so on
        """
        pass

    @abstractmethod
    def get_exploration_rate(self) -> float:
        """
        Retrieving the current exploration rate (epsilon)
        
        :return:
            the current exploration rate
        """
