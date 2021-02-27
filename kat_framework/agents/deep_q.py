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
from kat_framework.config.config_props import AgentConfigurationProperty
from kat_framework.agents.base import DiscreteAgent
from kat_framework.core.descriptors import TensorDescriptor
from kat_api import INetwork, IAgent, ITensorDescriptor, IReplayMemory, IReadOnlyMemory
from kat_api import NetworkInputType, IState
from kat_typing import TrainLoss
from overrides import overrides
from typing import Collection
import numpy as np


class QAgent(DiscreteAgent, IAgent):
    """
    Deep-Q learning Agent implementation.
    """
    # protected members

    _network_synchronization_frequency: int = 0
    _replay_memory: IReplayMemory = None
    _memory_access: IReadOnlyMemory = None

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        super(QAgent, self).__init__()

    @overrides
    def init(self,
             observation_space_desc: Collection[ITensorDescriptor],
             action_space_descriptor: ITensorDescriptor) -> None:
        """
        Object initialization.

        #see: IAgent.init(observation_space_desc: Collection[ITensorDescriptor],
                          action_space_desc: ITensorDescriptor)
        """
        super(QAgent, self).init(observation_space_desc=observation_space_desc,
                                 action_space_descriptor=action_space_descriptor)
        self._replay_memory = self._build_replay_memory()
        self._network = self._build_network()
        self._initialized = True

    @overrides
    def store_transition(self, state: IState) -> str:
        """
        Stores a transition to the associated replay memory.

        #see IAgent.store_transition(self, state: IState) -> str:
        """
        if state is None:
            raise ValueError("No state specified.")
        if self._frame_stacking_enabled and NetworkInputType.IMG == self._input_observation_type:
            # stacked last frame is currently in the buffer
            processed_s1_state = np.squeeze(np.stack(self._frame_buffer, axis=2), axis=-1)
            # emulate the transitioned state with a buffer copy
            processed_s2_state = self._pre_process_data(state.get_transitioned_observation())
            local_frame_buffer = self._frame_buffer.copy()
            local_frame_buffer.append(processed_s2_state)
            processed_s2_state = np.squeeze(np.stack(local_frame_buffer, axis=2), axis=-1)
            del local_frame_buffer
        else:
            processed_s1_state = self._pre_process_data(state.get_observation())
            processed_s2_state = self._pre_process_data(state.get_transitioned_observation())
        action_idx = self._action_space.index(state.get_transition())
        reward = state.get_reward()
        is_end_state = state.is_end_state()
        return self._replay_memory.add_transition(
            processed_s1_state,
            action_idx,
            processed_s2_state,
            reward,
            is_end_state)

    # protected member functions

    @overrides
    def _load_configuration(self):
        """
        Loads necessary configurations.
        """
        super(QAgent, self)._load_configuration()
        self._network_synchronization_frequency = self._config_handler.get_config_property(
            AgentConfigurationProperty.NETWORK_SYNCHRONIZATION_FREQUENCY,
            AgentConfigurationProperty.NETWORK_SYNCHRONIZATION_FREQUENCY.prop_type)

    @overrides
    def _train(self) -> TrainLoss:
        """
        Performs a training step on the specified batch.

        :return:
            current train loss
        """
        return self._network.train_batch(self._current_episode, self._current_step)

    def _build_memory_access(self) -> None:
        """
        Helper function for building readonly memory.
        """
        if self._memory_access is None:
            self._memory_access = KatherineApplication.get_application_factory().build_memory_access()
            self._memory_access.init(self._replay_memory)

    def _build_network(self) -> INetwork:
        """
        Helper function for building the network.

        :returns
            an initialized `INetwork` instance, based on the current configuration
        """
        self._build_memory_access()
        network = KatherineApplication.get_application_factory().build_network()
        network.init(replay_memory_access=self._memory_access,
                     input_descriptor=self._network_input_spec,
                     output_descriptor=self._network_output_spec,
                     is_distribution_enabled=self._distributed_learning_enabled)
        return network

    def _build_replay_memory(self) -> IReplayMemory:
        """
        Helper function for building the replay memory.

        :returns
            an initialized `IReplayMemory` instance, based on the current configuration
        """
        memory_max_size = self._config_handler.get_config_property(
            AgentConfigurationProperty.MEMORY_MAX_SIZE,
            AgentConfigurationProperty.MEMORY_MAX_SIZE.prop_type)
        memory_tensor_spec = \
            (TensorDescriptor('s1_states',
                              self._input_observation_dtype,
                              (memory_max_size, *self._frame_buffer_output_shape)),
             TensorDescriptor('action_ids', self._action_space_desc.get_data_type(), (memory_max_size,)),
             TensorDescriptor('s2_states',
                              self._input_observation_dtype,
                              (memory_max_size, *self._frame_buffer_output_shape)),
             TensorDescriptor('rewards', np.float32, (memory_max_size, )),
             TensorDescriptor('terminals', np.bool, (memory_max_size, )))
        memory = KatherineApplication.get_application_factory().build_memory()
        memory.init(memory_tensor_spec)
        return memory


class DQAgent(QAgent, IAgent):
    """
    Double Deep-Q learning Agent implementation.
    """

    # protected members

    _target_network: INetwork

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        super(DQAgent, self).__init__()

    @overrides
    def init(self,
             observation_space_desc: Collection[ITensorDescriptor],
             action_space_descriptor: ITensorDescriptor) -> None:
        """
        Object initialization.

        #see: IAgent.init(observation_space_desc: Collection[ITensorDescriptor],
                          action_space_desc: ITensorDescriptor)
        """
        super(DQAgent, self).init(observation_space_desc=observation_space_desc,
                                  action_space_descriptor=action_space_descriptor)
        self._target_network = self._build_network()
        self._inject_target_network()

    # protected member functions

    def _update_networks(self) -> None:
        """
        Helper function for synchronizing network weights.
        """
        if self._target_network is not None:
            self._target_network.set_weights(self._network.get_weights())

    @overrides
    def _train(self) -> TrainLoss:
        """
        Performs a training step on the specified batch.

        :return:
            current train loss
        """
        if self._current_step % self._network_synchronization_frequency == 0:
            self._update_networks()
        return super(DQAgent, self)._train()

    def _inject_target_network(self) -> None:
        """
        Helper function for reinitialize the evaluation network with the
        specified target network instance.
        """
        self._network.init(replay_memory_access=self._memory_access,
                           input_descriptor=self._network_input_spec,
                           output_descriptor=self._network_output_spec,
                           is_distribution_enabled=self._distributed_learning_enabled,
                           strategy=self._target_network.get_distribution_strategy(),
                           target_network=self._target_network)
