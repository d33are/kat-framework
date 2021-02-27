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
from typing import Optional, Tuple
from kat_typing import Tensor, IterableDataset


class IReplayMemory(metaclass=ABCMeta):
    """
    Common interface for replay buffers.

    Reinforcement learning algorithms use replay buffers to store experiences during execution.
    During training, replay buffers are sampling experiences to replay the agent's experience.

    A traditional implementation has 5 buffers:
        * originator states buffer
        * actions buffer
        * transitioned states buffer
        * rewards buffer
        * terminal states indicator buffer
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
                hasattr(subclass, 'get_number_of_frames') and
                callable(subclass.get_number_of_frames) and
                hasattr(subclass, 'add_transition') and
                callable(subclass.add_transition) and
                hasattr(subclass, 'update_transition_reward') and
                callable(subclass.update_transition_reward) and
                hasattr(subclass, 'get_sample') and
                callable(subclass.get_sample) and
                hasattr(subclass, 'as_iterable_dataset') and
                callable(subclass.as_iterable_dataset) and
                hasattr(subclass, 'get_all') and
                callable(subclass.get_all) and
                hasattr(subclass, 'reset') and
                callable(subclass.reset) or
                NotImplemented)

    @abstractmethod
    def init(self, buffer_spec: Tuple) -> None:
        """
        We're assuming that replay memories gonna' be instantiated through factory interfaces. Based on that
        knowledge, an ideal Agent implementation has a "no args" constructor, and a separated initialization
        method. (object instantiation is _NOT_ initialization)

        :param buffer_spec:
            tensor descriptors for each buffer
        """
        pass

    @abstractmethod
    def get_number_of_frames(self) -> int:
        """
        Total number of experiences stored currently in the memory.

        :returns:
            total number of experiences as integer
        """
        pass

    @abstractmethod
    def add_transition(self,
                       s1_state: Tensor,
                       action_idx: int,
                       s2_state: Tensor,
                       reward: float,
                       is_end_state: bool) -> str:
        """
        Adds a transition to the buffer.

        :returns
            given unique id of the stored experience
        """
        pass

    @abstractmethod
    def get_sample(self, sample_size: int) -> Tuple[Tensor,
                                                    Tensor,
                                                    Tensor,
                                                    Tensor,
                                                    Tensor]:
        """
        Sampling the buffer with the specified batch size.

        :param sample_size:
            sample size
        :return:
            A tuple of 5 with `sample_size` length samples from each buffer respectively.
            so, the expected tensor shapes are:
                ([sample_size,...], [sample_size,...], [sample_size,...], [sample_size,...], [sample_size,...])
        """
        pass

    @abstractmethod
    def as_iterable_dataset(self, input_context: Optional[object] = None) -> IterableDataset:
        """
        Samples the replay buffer, and returns it as an `Iterable`. Batch size is not given, the
        implementation is responsible for the Sharding, Batching, etc...

        :param input_context:
            context manager variables (Optional)
        :returns:
            an iterable dataset of samples from the buffer
        """
        pass

    @abstractmethod
    def get_all(self) -> Tuple[Tensor,
                               Tensor,
                               Tensor,
                               Tensor,
                               Tensor]:
        """
        Retrieving all content from the buffers.

        :return:
            A tuple of 5 with `max_size` length samples from each buffer respectively.
            so, the expected tensor shapes are:
                ([max_size,...], [max_size,...], [max_size,...], [max_size,...], [max_size,...])
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Clearing all buffers, and resets the buffer indexes. It is not necessary to "delete" all items in the buffer,
        in most cases index reset is enough.
        """
        pass


class IReadOnlyMemory(metaclass=ABCMeta):
    """
    Interface for readonly replay buffer adapters.

    In some cases we might provide data from the replay buffers, but only for read.
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
                hasattr(subclass, 'as_iterable_dataset') and
                callable(subclass.as_iterable_dataset) or
                NotImplemented)

    @abstractmethod
    def init(self, replay_memory: IReplayMemory):
        """
        :param replay_memory:
        :return:
        """
        pass

    @abstractmethod
    def as_iterable_dataset(self, input_context: Optional[object] = None) -> IterableDataset:
        """
        Samples the replay buffer, and returns it as an `Iterable`. Batch size is not given, the
        implementation is responsible for the Sharding, Batching, etc...

        :param input_context:
            context manager variables (Optional)
        :returns:
            an iterable dataset of samples from the buffer
        """
        pass
