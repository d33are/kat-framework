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

from kat_api import TraceableMetric
from kat_typing import MetricData


def build_metric_from_metadata(metric_metadata: TraceableMetric, subclass_of: type) -> MetricData:
    """
    Builds metric data based on the specified descriptor.

    :param metric_metadata:
        descriptor of the metric data
    :param subclass_of:
        subclass check expression
    :return:
        built metric based on the descriptor
    """
    if metric_metadata is None:
        raise ValueError("No metadata specified")
    if metric_metadata.metric_clazz is None:
        metric = metric_metadata.data_type
    else:
        if not issubclass(metric_metadata.metric_clazz, subclass_of):
            raise TypeError(
                "{} is not a subclass of tf.keras.metrics.Metric".format(metric_metadata.metric_clazz))
        metric = metric_metadata.metric_clazz(metric_metadata.label, metric_metadata.data_type)
    return metric
