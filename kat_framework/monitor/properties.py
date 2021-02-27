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

from kat_api import TraceableMetric, MetricType
import tensorflow as tf


class KatMetrics(TraceableMetric):
    """
    Basic metrics, which are traceable.

    (name, data_type, class, metric_type)
    """
    # Tensorflow loss mean
    TENSORFLOW_TRAIN_LOSS_MEAN = ("train_loss_mean", tf.float32, tf.keras.metrics.Mean, MetricType.SCALAR)
    # Tensorflow exploration rate
    TENSORFLOW_AGENT_EXPLORATION_RATE = ("agent_exploration_rate", tf.float32, None, MetricType.SCALAR)
    # Tensorflow agent total score
    TENSORFLOW_AGENT_TOTAL_SCORE = ("agent_total_score", tf.float32, None, MetricType.SCALAR)
