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
from kat_api import NetworkInputType, IAgent, ITensorDescriptor, IReplayMemory
from kat_api import IState
from kat_typing import TrainLoss
from overrides import overrides
from typing import Collection
import numpy as np


class RandomChoiceAgent(DiscreteAgent, IAgent):
    """
    Random Agent implementation. (for testing)

    (Pseudo)Randomly chooses an action from the given action space.
    """

    # protected members

    _replay_memory: IReplayMemory = None
    _network_input_spec: ITensorDescriptor = None
    _network_output_spec: ITensorDescriptor = None

    # public member function

    def __init__(self):
        """
        Default constructor.
        """
        super(RandomChoiceAgent, self).__init__()

    @overrides
    def init(self,
             observation_space_desc: Collection[ITensorDescriptor],
             action_space_descriptor: ITensorDescriptor) -> None:
        """
        Object initialization.

        #see: IAgent.init(observation_space_desc: Collection[ITensorDescriptor],
                          action_space_desc: ITensorDescriptor)
        """
        super(RandomChoiceAgent, self).init(observation_space_desc, action_space_descriptor)
        self._network_input_spec = TensorDescriptor(
            self._input_observation_name,
            self._input_observation_dtype,
            self._frame_buffer_output_shape,
            self._input_observation_type)
        self._network_output_spec = TensorDescriptor(
            "network_output",
            self._input_observation_dtype,
            self._action_space_desc.get_tensor_shape(),
            NetworkInputType.NONE)
        self._network = KatherineApplication.get_application_factory().build_network()
        self._network.init(output_descriptor=self._network_output_spec, input_descriptor=self._network_input_spec)
        self._replay_memory = self._build_replay_memory()

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

    # public member function

    @overrides
    def _train(self) -> TrainLoss:
        """
        Performs a training step on the specified batch.

        :return:
            current train loss
        """
        return self._network.train_batch()

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

