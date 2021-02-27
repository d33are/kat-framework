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

from abc import ABCMeta, abstractmethod
from kat_typing import MetricData, DistributionStrategy
from typing import Optional
from enum import Enum


class TraceableMetric(Enum):
    """
    Abstract base enum for metrics property descriptor.
    """
    def __new__(cls, label, data_type, metric_clazz, metric_type):
        obj = object.__new__(cls)
        obj.label = label
        obj.data_type = data_type
        obj.metric_clazz = metric_clazz
        obj.metric_type = metric_type
        obj._value_ = label
        return obj


class MetricType(Enum):
    """
    Abstract base enum for metric types.
    """
    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    """
    Basic possible metric types.
    """
    SCALAR = "scalar"
    IMAGE = "image"
    HYPER_PARAMETER = "hyper_parameter"
    FAIRNESS_INDICATOR = "fairness_indicator"
    EMBEDDING = "embedding"


class IMetricTracer(metaclass=ABCMeta):
    """
    Common interface for metric tracers.

    Metric tracers are able to store, and update `TraceableMetric` data during
    execution/training. Assuming in-memory cache based implementations with a
    `flush` functionality. (flush to disk, database, etc...)

    Traceable metrics can be: scalars, images, hyper parameters, fairness indicators, or embeddings.
    """

    @classmethod
    def __subclasshook__(cls, subclass: object):
        """Checks the class' expected behavior as a formal python interface.

        :param subclass:
            class to be checked

        :return:
            True if the object is a "real implementation" of this interface,
            otherwise False.
        """
        return (hasattr(subclass, 'is_initialized') and
                callable(subclass.is_initialized) and
                hasattr(subclass, 'start_profiler') and
                callable(subclass.start_profiler) and
                hasattr(subclass, 'stop_profiler') and
                callable(subclass.stop_profiler) and
                hasattr(subclass, 'flush_metrics') and
                callable(subclass.flush_metrics) and
                hasattr(subclass, 'update_metric') and
                callable(subclass.update_metric) or
                NotImplemented)

    @abstractmethod
    def init(self,
             distribution_strategy: Optional[DistributionStrategy] = None):
        """
        We're assuming that metric tracers gonna' be instantiated through factory interfaces. Based on that
        knowledge, an ideal tracer implementation has a "no args" constructor, and a separated initialization
        method. (object instantiation is _NOT_ initialization)

        :param distribution_strategy (Optional)
            Modern ML APIs can distribute training across multiple GPUs, multiple machines. It means
            the implementation must able to "know" that it is training on a single machine or on a cluster.
        """
        pass

    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Tracer's initialization state indicator.

        :returns
            True if the "init" method was called, otherwise false.
        """
        pass

    @abstractmethod
    def start_profiler(self):
        """
        Starts the GPU profiler, if any implementation is available.
        For example `cupti64.dll` for nvidia GPU drivers.
        """
        pass

    @abstractmethod
    def stop_profiler(self):
        """
        Stops the profiler and flush the captured data. In case of nvidia GPU-s the profiler is
        implemented as an in-memory profiler with a fix sized buffer.
        """
        pass

    @abstractmethod
    def flush_metrics(self, epoch_number: int) -> None:
        """
        Flush the current values of the metrics to a persistent storage. (disk, database, etc...)

        :param epoch_number:
            given epoch number where the flush was called
        """
        pass

    @abstractmethod
    def update_metric(self, metadata: TraceableMetric, data: MetricData) -> None:
        """
        Updates the specified metric with the specified new data.

        :param metadata:
            metric descriptor
        :param data:
            metric data
        """
        pass
