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

from kat_tensorflow.clusters.grpc import ParameterServerCluster
from kat_api import ITensorDescriptor
from kat_typing import TrainLoss, Tensor, Policy, DistributionStrategy, Activation
from kat_framework import NetworkConfigurationProperty, Network
from overrides import overrides
from abc import abstractmethod
from typing import Optional
from tensorflow.python.distribute.distribute_lib import _DefaultDistributionStrategy
import tensorflow as tf
import os


class TensorflowNetwork(Network):
    """
    Abstract base network class for Tensorflow based neural network
    implementations.
    """

    # protected members

    _is_distribution_enabled: bool = False
    _optimizer: tf.keras.optimizers.Optimizer = None
    _strategy: tf.distribute.Strategy = None
    _coordinator: tf.distribute.experimental.coordinator.ClusterCoordinator = None
    _restore_model_path: str = None
    _restore_checkpoint_path: str = None
    _learning_rate: float = 0.0
    _conv_layer_params: list = None
    _fc_layer_params: list = None
    _number_of_actions: int = 0
    _initialized: bool = False

    # public member functions

    def __init__(self, name: Optional[str] = None):
        """
        Default constructor.
        """
        super(TensorflowNetwork, self).__init__(name=name)

    @property
    def initialized(self):
        """
        Initialization flag.
        """
        return self._initialized

    def init(self,
             output_descriptor: ITensorDescriptor,
             input_descriptor: Optional[ITensorDescriptor] = None,
             is_distribution_enabled: Optional[bool] = False,
             strategy: Optional[DistributionStrategy] = None):
        """
        Object initialization.

        # see: INetwork.init( ... )
        """
        super(TensorflowNetwork, self).init(input_descriptor=input_descriptor, output_descriptor=output_descriptor)
        self._is_distribution_enabled = is_distribution_enabled
        self._number_of_actions = self._network_output_descriptor.get_tensor_shape()[0]
        if self._is_distribution_enabled:
            # "Set the environment variable to allow reporting worker and ps failure to the
            # coordinator. This is a workaround and won't be necessary in the future."
            os.environ["GRPC_FAIL_FAST"] = "use_caller"
            if strategy is None:
                cluster = ParameterServerCluster()
                self._strategy = tf.distribute.experimental.ParameterServerStrategy(
                    cluster_resolver=cluster.get_cluster_resolver(),
                    variable_partitioner=cluster.get_partitioner())
            else:
                self._strategy = strategy
            self._coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(self._strategy)
        else:
            if strategy is None:
                self._strategy = tf.distribute.get_strategy()
            else:
                self._strategy = strategy
        with self._strategy.scope():
            if self._restore_model_path is not None:
                self._network_model = self._serializer.restore_model(self._restore_model_path)
            else:
                self._network_model = self._create_model()
            if self._restore_checkpoint_path is not None:
                self._network_model = self._serializer.restore_checkpoint(
                    self._network_model, self._restore_checkpoint_path)
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        self._initialized = True

    @overrides
    def train_batch(self, current_episode: Optional[int] = 0, current_step: Optional[int] = 0) -> TrainLoss:
        """
        Object initialization.

        # see: Network.train_batch(current_episode, current_step)
        """
        if isinstance(self._strategy,
                      (tf.distribute.experimental.ParameterServerStrategy,
                       tf.distribute.experimental.CentralStorageStrategy)):
            loss = self._coordinator.schedule(self._train_batch).fetch()
        elif isinstance(self._strategy, _DefaultDistributionStrategy):
            loss = self._train_batch()
        else:
            raise RuntimeError("Strategy {} not supported.".format(type(self._strategy)))
        super(TensorflowNetwork, self).train_batch(current_episode, current_step)
        return loss

    def get_weights(self):
        """
        # see: Network.get_weights()
        """
        return self._network_model.get_weights()

    def get_distribution_strategy(self) -> DistributionStrategy:
        """
        # see: Network.get_distribution_strategy()
        """
        if self.initialized:
            return self._strategy
        return None

    # protected member functions

    @overrides
    def _load_configuration(self):
        """
        Loads the necessary configuration.
        """
        super(TensorflowNetwork, self)._load_configuration()
        self._restore_model_path = self._config_handler.get_config_property(
            NetworkConfigurationProperty.RESTORE_MODEL_FROM,
            NetworkConfigurationProperty.RESTORE_MODEL_FROM.prop_type)
        self._restore_checkpoint_path = self._config_handler.get_config_property(
            NetworkConfigurationProperty.RESTORE_CHECKPOINT_FROM,
            NetworkConfigurationProperty.RESTORE_CHECKPOINT_FROM.prop_type)
        self._learning_rate = self._config_handler.get_config_property(
            NetworkConfigurationProperty.OPTIMIZER_LEARNING_RATE,
            NetworkConfigurationProperty.OPTIMIZER_LEARNING_RATE.prop_type)
        self._conv_layer_params = self._config_handler.get_config_property(
            NetworkConfigurationProperty.CONVOLUTION_PARAMETERS,
            NetworkConfigurationProperty.CONVOLUTION_PARAMETERS.prop_type)
        self._fc_layer_params = self._config_handler.get_config_property(
            NetworkConfigurationProperty.FULLY_CONNECTED_PARAMETERS,
            NetworkConfigurationProperty.FULLY_CONNECTED_PARAMETERS.prop_type)

    def _build_encoder(self, input_layer: tf.keras.layers.Layer, activation_fn: Activation) -> tf.keras.layers.Layer:
        """
        Building encoder layers based on configuration.

        :param input_layer:
            input keras layer
        :param activation_fn:
            activation function to use
        :return:
            output layer of the encoder
        """
        x = input_layer
        if self._conv_layer_params is not None:
            num_of_conv = len(self._conv_layer_params)
            for i, config in enumerate(self._conv_layer_params):
                filters, kernel_size, strides = config
                y = tf.keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        activation=activation_fn)(x)
                if i < (num_of_conv - 1):
                    y = tf.keras.layers.MaxPooling2D(padding='valid')(y)
                x = y
            x = tf.keras.layers.GlobalMaxPool2D()(x)
        if self._fc_layer_params is not None:
            for num_units in self._fc_layer_params:
                y = tf.keras.layers.Dense(
                    num_units,
                    activation=activation_fn,
                    kernel_regularizer=None)(x)
                x = y
        return x

    @abstractmethod
    def _create_model(self) -> tf.keras.Model:
        """
        Derived classes must implement the model creation.

        self._network_model = tf.Sequential( ... )
        """
        pass

    @overrides
    @abstractmethod
    def _predict(self, input_tensor: Tensor) -> Policy:
        """
        # see : Network._predict(input_tensor)
        """
        pass

    @abstractmethod
    def _train_batch(self) -> TrainLoss:
        """
        # see : Network._train_batch()
        """
        pass
