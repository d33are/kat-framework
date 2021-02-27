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

from kat_api import IReplayMemory
from kat_typing import IterableDataset
from kat_framework import UniformMemory, KatherineApplication, KatConfigurationProperty
from overrides import overrides
import tensorflow as tf


class TensorflowUniformMemory(UniformMemory, IReplayMemory):
    """
    Tensorflow dataset based uniform memory implementation.
    """

    # protected members

    _batch_size: int = 0

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        super(TensorflowUniformMemory, self).__init__()
        self._load_configuration()

    def data_generator(self) -> tuple:
        """
        Data generator for the tensorflow dataset.

        :return:
            a tuple (initiator_states, action_ids, transitioned_states, rewards, end_state_factors)
            of arrays, with batch dimension = _batch_size
        """
        while True:
            initiator_states, action_ids, transitioned_states, rewards, end_state_factors = \
                self.get_sample(self._batch_size)
            yield initiator_states, action_ids, transitioned_states, rewards, end_state_factors

    @overrides
    def as_iterable_dataset(self, input_context: object = None) -> IterableDataset:
        """
        Builds the tensorflow dataset.

        :param input_context:
            input context (tensorflow passes it)
        :return:
            tf.data.Dataset from generator
        """
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                tf.TensorSpec(shape=(self._batch_size, *self._s1_states_spec.get_tensor_shape()),
                              dtype=self._s1_states_spec.get_data_type()),
                tf.TensorSpec(shape=(self._batch_size, *self._action_ids_spec.get_tensor_shape()),
                              dtype=self._action_ids_spec.get_data_type()),
                tf.TensorSpec(shape=(self._batch_size, *self._s2_states_spec.get_tensor_shape()),
                              dtype=self._s2_states_spec.get_data_type()),
                tf.TensorSpec(shape=(self._batch_size, *self._rewards_spec.get_tensor_shape()),
                              dtype=self._rewards_spec.get_data_type()),
                tf.TensorSpec(shape=(self._batch_size, *self._terminals_spec.get_tensor_shape()),
                              dtype=self._terminals_spec.get_data_type())))
        return dataset.prefetch(1)

    # protected member functions

    def _load_configuration(self) -> None:
        """
        Loads the configuration.
        """
        self._batch_size = KatherineApplication.get_application_config().get_config_property(
            KatConfigurationProperty.TRAIN_BATCH_SIZE,
            KatConfigurationProperty.TRAIN_BATCH_SIZE.prop_type
        )
