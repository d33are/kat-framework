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

import multiprocessing
import portpicker
import tensorflow as tf

NUM_WORKERS = 3
NUM_PS = 2


def create_in_process_tensorflow_cluster() -> tf.distribute.cluster_resolver.ClusterResolver:
    """
    Creates and starts local servers and returns the cluster_resolver.
    """
    worker_ports = [portpicker.pick_unused_port() for _ in range(NUM_WORKERS)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(NUM_PS)]

    cluster_dict = {"worker": ["localhost:%s" % port for port in worker_ports]}
    if NUM_PS > 0:
        cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

    cluster_spec = tf.train.ClusterSpec(cluster_dict)

    # Workers need some inter_ops threads to work properly.
    worker_config = tf.compat.v1.ConfigProto()
    if multiprocessing.cpu_count() < NUM_WORKERS + 1:
        worker_config.inter_op_parallelism_threads = NUM_WORKERS + 1

    for i in range(NUM_WORKERS):
        tf.distribute.Server(
            cluster_spec, job_name="worker", task_index=i, config=worker_config,
            protocol="grpc")

    for i in range(NUM_PS):
        tf.distribute.Server(
            cluster_spec, job_name="ps", task_index=i, protocol="grpc")

    return tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec, rpc_layer="grpc")


def create_in_process_tensorflow_fixed_partitioner() -> tf.distribute.experimental.partitioners.FixedShardsPartitioner:
    """
    Creates a FixedShardPartitioner.
    """
    return tf.distribute.experimental.partitioners.FixedShardsPartitioner(num_shards=NUM_PS)
