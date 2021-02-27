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

from typing import Collection, Tuple
from kat_framework.framework import KatherineApplication
from kat_framework.config.config_props import KatConfigurationProperty
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
import tensorflow as tf

WORKER = "worker"
PS = "ps"


class ParameterServerCluster:
    """
    Parameter server cluster for tensorflow based implementations.
    """

    # protected members

    _cluster_dict: dict = None
    _cluster_spec: tf.train.ClusterSpec = None
    _num_ps: int = 0

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        super(ParameterServerCluster, self).__init__()
        self._cluster_dict = {}
        workers = []
        p_servers = []
        cluster_list = ParameterServerCluster._load_configuration()
        for member in cluster_list:
            if WORKER == member[0]:
                workers.append("{}{}{}".format(str(member[1]), ":", str(member[2])))
            elif PS == member[0]:
                self._num_ps += 1
                p_servers.append("{}{}{}".format(str(member[1]), ":", str(member[2])))
            else:
                raise ValueError("Unknown server type: {}".format(member[0]))
        self._cluster_dict[WORKER] = workers
        self._cluster_dict[PS] = p_servers

    def get_cluster_resolver(self) -> SimpleClusterResolver:
        """
        Builds a simple tensorflow cluster resolver.

        :return:
            cluster resolver instance
        """
        if self._cluster_spec is None:
            self._cluster_spec = tf.train.ClusterSpec(self._cluster_dict)
        return tf.distribute.cluster_resolver.SimpleClusterResolver(
            self._cluster_spec, rpc_layer="grpc")

    def get_partitioner(self) -> \
            tf.distribute.experimental.partitioners.FixedShardsPartitioner:
        """
        Builds a fixed shards partitioner, based on the number of parameter servers.
        """
        return tf.distribute.experimental.partitioners.FixedShardsPartitioner(num_shards=self._num_ps)

    # protected member functions

    @staticmethod
    def _load_configuration() -> Collection[Tuple]:
        """
        Loads the server configurations.

        :return:
            a list of tuples with server configs
        """
        return KatherineApplication.get_application_config().get_config_property(
            KatConfigurationProperty.CLUSTER_INFO,
            KatConfigurationProperty.CLUSTER_INFO.prop_type)
