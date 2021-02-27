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

from kat_tensorflow.networks.base import TensorflowNetwork
from kat_api import INetwork, ITensorDescriptor, IReadOnlyMemory
from kat_typing import TrainLoss, Tensor, Policy, DistributionStrategy, IterableDataset
from kat_framework import NetworkConfigurationProperty
from typing import Optional
from overrides import overrides
import tensorflow as tf


class QNetwork(TensorflowNetwork, INetwork):
    """
    Deep Q learning `INetwork` implementation.

    We feed in the state, pass that through several hidden layers and then output the Q-values.
    """

    # protected members

    _target_network: INetwork = None
    _discount_factor: float = 0.0
    _per_worker_dataset: IterableDataset = None
    _per_worker_iterator: iter = None
    _replay_memory: IReadOnlyMemory = None

    # public member functions

    def __init__(self, name: str = 'QNetwork'):
        """
        Default constructor
        """
        super(QNetwork, self).__init__(name=name)

    @overrides
    def init(self,
             output_descriptor: ITensorDescriptor,
             replay_memory_access: Optional[IReadOnlyMemory] = None,
             input_descriptor: Optional[ITensorDescriptor] = None,
             is_distribution_enabled: Optional[bool] = False,
             strategy: Optional[DistributionStrategy] = None,
             target_network: Optional = None):
        """
        Object initialization.

        # see : INetwork.init( ... )
        """
        super(QNetwork, self).init(input_descriptor=input_descriptor,
                                   output_descriptor=output_descriptor,
                                   is_distribution_enabled=is_distribution_enabled,
                                   strategy=strategy)
        self._replay_memory = replay_memory_access
        if self._is_distribution_enabled:
            self._per_worker_dataset = self._coordinator.create_per_worker_dataset(self._distributed_dataset_fn)
            self._per_worker_iterator = iter(self._per_worker_dataset)
        else:
            self._per_worker_dataset = self._simple_dataset_fn()
            self._per_worker_iterator = iter(self._per_worker_dataset)
        self._target_network = target_network
        self._initialized = True

    @overrides
    def set_weights(self, weights: object):
        """
        # see : INetwork.set_weights(weights)
        """
        if weights is None:
            raise ValueError("No weights specified")
        self._network_model.set_weights(weights)

    # protected member functions

    @overrides
    def _load_configuration(self):
        """
        Loads the necessary configuration.
        """
        super(QNetwork, self)._load_configuration()
        self._discount_factor = self._config_handler.get_config_property(
            NetworkConfigurationProperty.REWARD_DISCOUNT_FACTOR,
            NetworkConfigurationProperty.REWARD_DISCOUNT_FACTOR.prop_type)

    def _create_model(self) -> tf.keras.Model:
        """
        Builds a basic Q network.

        Encoder + "number of actions" dense layer
        """
        input_layer = tf.keras.Input(shape=self._input_descriptor.get_tensor_shape())
        encoder = self._build_encoder(input_layer=input_layer, activation_fn=tf.keras.layers.LeakyReLU(alpha=0.001))
        output_layer = tf.keras.layers.Dense(self._number_of_actions, activation=None)(encoder)
        model = tf.keras.Model(
            inputs=[input_layer],
            outputs=[output_layer])
        return model

    def _simple_dataset_fn(self):
        """
        Retrieves a batch in an iterable format.
        """
        return self._replay_memory.as_iterable_dataset()

    @tf.function
    def _distributed_dataset_fn(self) -> IterableDataset:
        """
        Retrieves a batch in an iterable format. (for distributed learning)
        """
        return self._strategy.distribute_datasets_from_function(self._replay_memory.as_iterable_dataset)

    @tf.function
    def _predict(self, input_tensor: Tensor) -> Policy:
        """
        # see : Network._predict(input)
        """
        return self._network_model(input_tensor)

    @tf.function
    def _train_batch(self) -> TrainLoss:
        """
        # see : Network._train_batch()
        """

        def replica_fn(iterator, d_factor):
            initiator_states, action_ids, transitioned_states, rewards, end_state_factors = next(iterator)
            if self._target_network is not None:
                q_transitioned = tf.reduce_max(self._target_network.predict(transitioned_states), axis=-1)
            else:
                q_transitioned = tf.reduce_max(self._network_model(transitioned_states), axis=-1)
            q_target = tf.where(end_state_factors, rewards, rewards + d_factor * q_transitioned)
            with tf.GradientTape() as tape:
                q_evaluation = self._network_model(initiator_states, training=True)
                q_prediction = tf.math.reduce_sum(q_evaluation * tf.one_hot(action_ids, self._number_of_actions),
                                                  axis=1)
                loss = tf.reduce_mean(tf.square(q_prediction - q_target))
            variables = self._network_model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self._optimizer.apply_gradients(zip(gradients, variables))
            return loss

        losses = self._strategy.run(replica_fn, args=(self._per_worker_iterator, self._discount_factor))
        return self._strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)


class DuelingQNetwork(QNetwork, INetwork):
    """
    Dueling Deep Q learning `INetwork` implementation.

    Very similar to the traditional Q network, but the dense layer is
    separated to a Value Layer and an Advantage Layer. Combining the two
    layer will gives the output.
    """

    def __init__(self, name: str = 'DuelingQNetwork'):
        """
        Default constructor.
        """
        super(DuelingQNetwork, self).__init__(name)

    @overrides
    def _create_model(self) -> tf.keras.Model:
        """
        Builds a basic Dueling Q network.

        combined output = A + (V - mean(V))
        """
        input_layer = tf.keras.Input(shape=self._input_descriptor.get_tensor_shape())
        encoder = self._build_encoder(input_layer=input_layer, activation_fn=tf.keras.layers.LeakyReLU(alpha=0.001))
        v = tf.keras.layers.Dense(1, activation=None)(encoder)
        v_normalized = tf.keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))(v)
        a = tf.keras.layers.Dense(self._number_of_actions, activation=None)(encoder)
        output_layer = tf.keras.layers.Add()([a, v_normalized])
        model = tf.keras.Model(
            inputs=[input_layer],
            outputs=[output_layer])
        return model

