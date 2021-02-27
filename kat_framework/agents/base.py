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
from kat_framework.config.config_props import AgentConfigurationProperty, KatConfigurationProperty
from kat_framework.core.descriptors import TensorDescriptor
from kat_framework.util import logger, tensors
from kat_api import ITensorDescriptor, IState, IObservation, INetwork, IConfigurationHandler, NetworkInputType
from kat_typing import Action, TrainLoss, Tensor, DistributionStrategy
from typing import Collection, List
from abc import abstractmethod, ABCMeta
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from overrides import overrides
from logging import Logger
import itertools as it
import numpy as np
import random

UNCHECKED_WARN_MSG = "Unchecked input will be fed to the network, assuming unexpected behavior."


class BaseAgent(metaclass=ABCMeta):
    """
    Abstract base class (ABC) for `IAgent` implementations.

    Main responsibilities:
        * image preprocessing
        * image stacking
        * exploration rate calculation
        * input/output specification checks
    """

    # class logger
    _log: Logger = None

    # protected members
    _config_handler: IConfigurationHandler = None
    _screen_size: tuple = None
    _screen_channels: int = 0
    _convert_to_monochrome: bool = False
    _frame_stacking_enabled: bool = False
    _number_of_stacked_frames: int = 0
    _frame_buffer: deque = None
    _frame_buffer_output_shape: tuple = None
    _input_tensor_shape: tuple = None
    _observation_space_desc: Collection[ITensorDescriptor] = None
    _action_space_desc: ITensorDescriptor = None
    _input_observation_desc: ITensorDescriptor = None
    _input_observation_name: str = None
    _input_observation_type: NetworkInputType = None
    _input_observation_dtype: type = None
    _network: INetwork = None
    _network_input_spec: ITensorDescriptor = None
    _network_output_spec: ITensorDescriptor = None
    _train_batch_size: int = 0
    _distributed_learning_enabled: bool = False
    _initial_epsilon: float = 0.0
    _final_epsilon: float = 0.0
    _constant_exploration: float = 0.0
    _decaying_exploration: float = 0.0
    _max_episodes: int = 0
    _epsilon: float = 0
    _current_episode: int = 0
    _current_step: int = 0
    _max_observe_episodes: int = 0
    _initialized: bool = False

    # public members functions

    def __init__(self):
        """
        Default constructor.
        """
        super(BaseAgent, self).__init__()
        self._log = logger.get_logger(self.__class__.__name__)
        self._load_configuration()

    def init(self,
             observation_space_desc: Collection[ITensorDescriptor],
             action_space_descriptor: ITensorDescriptor) -> None:
        """
        Object initialization.

        #see: IAgent.init(observation_space_desc: Collection[ITensorDescriptor],
                          action_space_desc: ITensorDescriptor)
        """
        if action_space_descriptor is None:
            raise ValueError("No action space descriptor specified.")
        if observation_space_desc is None:
            raise ValueError("No observation space specified.")
        self._action_space_desc = action_space_descriptor
        self._observation_space_desc = observation_space_desc
        self._input_observation_desc = next(filter(lambda x: x.get_display_name() == self._input_observation_name,
                                                   self._observation_space_desc))
        self._input_observation_type = self._input_observation_desc.get_network_input_type()
        if NetworkInputType.RAM == self._input_observation_type:
            # assuming no preprocessing on RAM data
            self._input_tensor_shape = self._input_observation_desc.get_tensor_shape()
            self._input_observation_dtype = self._input_observation_desc.get_data_type()
            self._frame_buffer_output_shape = self._input_tensor_shape
        elif NetworkInputType.IMG == self._input_observation_type:
            # preprocessed (rescaled) dimensions
            self._input_tensor_shape = (*self._screen_size, self._screen_channels)
            self._input_observation_dtype = np.float32
            if self._frame_stacking_enabled:
                self._frame_buffer = deque(
                    [np.zeros(self._input_tensor_shape, dtype=self._input_observation_dtype)
                     for i in range(self._number_of_stacked_frames)], maxlen=self._number_of_stacked_frames)
                self._frame_buffer_output_shape = (
                    *self._screen_size, self._screen_channels * self._number_of_stacked_frames)
            else:
                self._frame_buffer_output_shape = self._input_tensor_shape
        else:
            raise RuntimeError("Unexpected input type: {}".format(str(self._input_observation_type)))
        self._initialized = True

    def tick(self, current_episode: int, current_step: int) -> None:
        """
        Handles per tick calls.

        #see: IAgent.tick(self, current_episode: int, current_step: int)
        """
        self._current_episode = current_episode
        self._current_step = current_step

    def take_action(self, game_state: IState) -> Action:
        """
        Takes an action from the action space, based on policy.

        #see: IAgent.take_action(self, game_state: IState)
        """
        observation = game_state.get_observation()
        if self._observation_space_desc is not None:
            if not tensors.check_same_observation_structure(observation, self._observation_space_desc):
                raise RuntimeError("Input structure vs observation mismatch.")
        else:
            self._log.warning(UNCHECKED_WARN_MSG)
        if observation is None:
            raise ValueError("No state specified.")
        processed_observation = self._pre_process_data(observation)
        processed_observation = self._stack_frames(processed_observation)
        self._epsilon = self._update_exploration_rate(game_state.get_state_id())
        return self._take_action(processed_observation)

    def train(self) -> TrainLoss:
        """
        Performs a train step on a specified batch.

        #see: IAgent.train(self)
        """
        if self._current_episode <= self._max_observe_episodes:
            return 0.0  # no loss
        return self._train()

    def is_initialized(self) -> bool:
        """
        Class initialization flag.

        #see: IAgent.is_initialized(self)
        """
        return self._initialized

    def get_distribution_strategy(self) -> DistributionStrategy:
        """
        Rerturns the distribution strategy.

        #see: IAgent.get_distribution_strategy(self)
        """
        if self.is_initialized():
            return self._network.get_distribution_strategy()

    def persist_model(self):
        """
        Persists the model on a specified storage.

        #see: IAgent.persist_model(self):
        """
        if self.is_initialized():
            self._network.persist_model()

    def get_exploration_rate(self) -> float:
        return self._epsilon

    # protected member functions

    @abstractmethod
    def _take_action(self, observation: Tensor) -> Action:
        """
        `take_action` method for derived classes, the public implementation of `take_action` is
        reserved for the base class.

        :param observation:
            preprocessed observation tensor
        :return:
            the action taken by the network
        """
        pass

    @abstractmethod
    def _train(self) -> TrainLoss:
        """
        `train` method for derived classes, the public implementation of `train` is
        reserved for the base class.

        :return:
            current loss value (train_batch)
        """
        pass

    @abstractmethod
    def _build_network_specs(self) -> None:
        """
        Helper function for building network descriptors.
        Derived classes should call it, the base is not calling it.
        """
        pass

    def _load_configuration(self):
        """
        Loads necessary configurations.
        """
        self._config_handler = KatherineApplication.get_application_config()
        if not self._config_handler:
            raise ValueError("No configuration handler specified.")
        self._train_batch_size = self._config_handler.get_config_property(
            KatConfigurationProperty.TRAIN_BATCH_SIZE,
            KatConfigurationProperty.TRAIN_BATCH_SIZE.prop_type)
        self._distributed_learning_enabled = self._config_handler.get_config_property(
            AgentConfigurationProperty.DISTRIBUTED_LEARNING_ENABLED,
            AgentConfigurationProperty.DISTRIBUTED_LEARNING_ENABLED.prop_type)
        self._convert_to_monochrome = self._config_handler.get_config_property(
            AgentConfigurationProperty.CONVERT_TO_MONOCHROME,
            AgentConfigurationProperty.CONVERT_TO_MONOCHROME.prop_type)
        self._screen_channels = self._config_handler.get_config_property(
            AgentConfigurationProperty.SCREEN_CHANNELS,
            AgentConfigurationProperty.SCREEN_CHANNELS.prop_type)
        self._frame_stacking_enabled = self._config_handler.get_config_property(
            AgentConfigurationProperty.FRAME_STACKING_ENABLED,
            AgentConfigurationProperty.FRAME_STACKING_ENABLED.prop_type)
        self._number_of_stacked_frames = self._config_handler.get_config_property(
            AgentConfigurationProperty.NUMBER_OF_STACKED_FRAMES,
            AgentConfigurationProperty.NUMBER_OF_STACKED_FRAMES.prop_type)
        self._screen_size = (
            self._config_handler.get_config_property(
                AgentConfigurationProperty.SCREEN_HEIGHT,
                AgentConfigurationProperty.SCREEN_HEIGHT.prop_type),
            self._config_handler.get_config_property(
                AgentConfigurationProperty.SCREEN_WEIGHT,
                AgentConfigurationProperty.SCREEN_WEIGHT.prop_type))
        self._input_observation_name = self._config_handler.get_config_property(
            AgentConfigurationProperty.INPUT_OBSERVATION_NAME,
            AgentConfigurationProperty.INPUT_OBSERVATION_NAME.prop_type)
        self._initial_epsilon = self._config_handler.get_config_property(
            AgentConfigurationProperty.INITIAL_EXPLORATION_RATE,
            AgentConfigurationProperty.INITIAL_EXPLORATION_RATE.prop_type)
        self._final_epsilon = self._config_handler.get_config_property(
            AgentConfigurationProperty.FINAL_EXPLORATION_RATE,
            AgentConfigurationProperty.FINAL_EXPLORATION_RATE.prop_type)
        self._constant_exploration = self._config_handler.get_config_property(
            AgentConfigurationProperty.CONSTANT_EXPLORATION_PERCENTAGE,
            AgentConfigurationProperty.CONSTANT_EXPLORATION_PERCENTAGE.prop_type)
        self._decaying_exploration = self._config_handler.get_config_property(
            AgentConfigurationProperty.DECAYING_EXPLORATION_PERCENTAGE,
            AgentConfigurationProperty.DECAYING_EXPLORATION_PERCENTAGE.prop_type)
        self._max_episodes = self._config_handler.get_config_property(
            KatConfigurationProperty.MAX_EPISODES,
            KatConfigurationProperty.MAX_EPISODES.prop_type)
        self._max_observe_episodes = self._config_handler.get_config_property(
            AgentConfigurationProperty.MAX_OBSERVE_EPISODES,
            AgentConfigurationProperty.MAX_OBSERVE_EPISODES.prop_type)

    def _pre_process_data(self, observation: IObservation) -> Tensor:
        """
        Preprocessing the screen buffer, based on self._screen_size.

        :returns
            resized and reshaped observation data based on app settings
        """
        if observation is None:
            return np.zeros(self._input_tensor_shape, dtype=self._input_observation_dtype)
        tensor = getattr(observation, self._input_observation_name)
        if NetworkInputType.IMG == self._input_observation_type:
            tensor = resize(tensor, self._screen_size)
            if self._convert_to_monochrome:
                tensor = rgb2gray(tensor)
            tensor = np.reshape(tensor, self._input_tensor_shape)
            tensor = tensor.astype(self._input_observation_dtype, copy=False)
        return tensor

    def _stack_frames(self, current_frame: Tensor) -> Tensor:
        """
        Persists the model on a specified storage.

        #see: IAgent.persist_model(self):
        """
        stacked_state = current_frame
        if self._frame_stacking_enabled and NetworkInputType.IMG == self._input_observation_type:
            self._frame_buffer.append(stacked_state)
            stacked_state = np.squeeze(np.stack(self._frame_buffer, axis=2), axis=-1)
        return stacked_state

    def _update_exploration_rate(self, current_episode) -> float:
        """
        Calculates the exploration rate curve, based on the current epoch.

        :returns
            updated exploration rate
        """
        constant_exp_episodes = self._constant_exploration * self._max_episodes
        decaying_exp_episodes = self._decaying_exploration * self._max_episodes
        delimiter = (decaying_exp_episodes - constant_exp_episodes) * (self._initial_epsilon - self._final_epsilon)
        if current_episode < constant_exp_episodes:
            # Constant Phase
            return self._initial_epsilon
        elif current_episode < decaying_exp_episodes:
            # Decaying Phase
            return self._initial_epsilon - (current_episode - constant_exp_episodes) / delimiter
        else:
            # Release phase
            return self._final_epsilon


class DiscreteAgent(BaseAgent):
    """
    Abstract base class (ABC) for IAgents with discrete action spaces.

    Main responsibilities:
        * building action space
        * taking actions based on input observations
        * epsilon greedy
    """

    # protected members
    _action_space: List[Action] = None
    _number_of_actions: int = 0
    _one_hot_encoded_action_space: bool = False

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        super(DiscreteAgent, self).__init__()

    @overrides
    def init(self,
             observation_space_desc: Collection[ITensorDescriptor],
             action_space_descriptor: ITensorDescriptor) -> None:
        """
        Object initialization.

        #see: IAgent.init(observation_space_desc: Collection[ITensorDescriptor], action_space_desc: ITensorDescriptor)
        """
        super(DiscreteAgent, self).init(observation_space_desc=observation_space_desc,
                                        action_space_descriptor=action_space_descriptor)
        self._action_space = self._build_action_space()
        self._number_of_actions = len(self._action_space)
        self._build_network_specs()
        self._initialized = True

    # protected member functions

    @abstractmethod
    def _train(self) -> TrainLoss:
        """
        # see : BaseAgent._train()
        """
        pass

    @overrides
    def _load_configuration(self):
        """
        Loads necessary configurations.

        #see BaseAgent._load_configuration()
        """
        super(DiscreteAgent, self)._load_configuration()
        self._one_hot_encoded_action_space = self._config_handler.get_config_property(
            AgentConfigurationProperty.ONE_HOT_ENCODED_ACTION_SPACE,
            AgentConfigurationProperty.ONE_HOT_ENCODED_ACTION_SPACE.prop_type)

    @overrides
    def _take_action(self, observation: Tensor) -> Action:
        """
        Takes an action, based on the current observation.

        #see BaseAgent._take_action()
        #see BaseAgent.take_action()
        """
        if np.random.rand() < self._epsilon:
            action_idx = random.randrange(self._number_of_actions)
        else:
            input_t = np.array([observation])
            policy = self._network.predict(input_t)
            if isinstance(policy, np.ndarray):
                action_idx = np.argmax(policy)
            else:
                action_idx = np.argmax(policy.numpy())
            if isinstance(action_idx, np.ndarray):
                action_idx = action_idx[0]
        return self._action_space[action_idx]

    def _build_action_space(self) -> List[Action]:
        """
        Building the Agent's action space based on the provided
        action space descriptors.

        :returns
            list of actions (discrete action space)
        """
        shape = self._action_space_desc.get_tensor_shape()
        dtype = self._action_space_desc.get_data_type()
        # assuming that a 1 dimensional integer tensor is a discrete action space
        if len(shape) == 1 and dtype == np.int32:
            n = shape[0]
            if self._one_hot_encoded_action_space:
                return [list(a) for a in it.product([0, 1], repeat=n)]
            else:
                return list([a for a in range(n)])
        else:
            raise ValueError("action descriptor is specifying a non discrete space")

    @overrides
    def _build_network_specs(self) -> None:
        """
        Helper function for building network descriptors.

        # see : BaseAgent._build_network_specs()
        """
        if self._network_input_spec is None:
            self._network_input_spec = TensorDescriptor(
                self._input_observation_name,
                self._input_observation_dtype,
                self._frame_buffer_output_shape,
                self._input_observation_type)
        if self._network_output_spec is None:
            self._network_output_spec = TensorDescriptor(
                "network_output",
                self._input_observation_dtype,
                (self._number_of_actions, ),
                NetworkInputType.NONE)
