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

from typing import Optional, Collection
from kat_api import IMetricTracer, TraceableMetric, MetricType, IConfigurationHandler
from kat_framework import KatherineApplication, TensorBoardConfigurationProperty, KatMetrics
from kat_framework.util import logger, fileio, metrics
from kat_typing import MetricData, DistributionStrategy
from overrides import overrides
from logging import DEBUG
import tensorflow as tf

WORKING_DIRECTORY_PREFIX = "metrics"


class TensorboardTracer(IMetricTracer):
    """
    Tensorflow based metrics collector implementation.

    # see : IMetricTracer
    """

    _log = logger.get_logger(__name__ + ".TensorboardTracer")
    config_handler: IConfigurationHandler = None
    enabled_metrics: Collection[KatMetrics] = None
    is_profiler_enabled: bool = False
    work_directory: str = None
    strategy: DistributionStrategy = None
    metrics: dict = None
    summary_writer: tf.summary.SummaryWriter = None
    initialized: bool = False

    def __init__(self):
        """
        Default constructor.
        """
        config_handler = KatherineApplication.get_application_config()
        self.enabled_metrics = config_handler.get_config_property(
            TensorBoardConfigurationProperty.ENABLED_NETWORK_METRICS,
            TensorBoardConfigurationProperty.ENABLED_NETWORK_METRICS.prop_type)
        self.is_profiler_enabled = config_handler.get_config_property(
            TensorBoardConfigurationProperty.GPU_PROFILER_ENABLED,
            TensorBoardConfigurationProperty.GPU_PROFILER_ENABLED.prop_type)
        self.work_directory = fileio.build_metrics_work_directory(WORKING_DIRECTORY_PREFIX)

    @overrides
    def is_initialized(self) -> bool:
        """
        Initialization flag.

        # see : IMetricTracer.is_initialized()
        """
        return self.initialized

    @overrides
    def init(self, distribution_strategy: Optional[DistributionStrategy] = None):
        """
        Object initialization.

        # see : IMetricTracer.init(distribution_strategy)
        """
        if distribution_strategy is None:
            self.strategy = tf.distribute.get_strategy()
        else:
            self.strategy = distribution_strategy
        with self.strategy.scope():
            self.metrics = self._build_metrics()
            self.summary_writer = tf.summary.create_file_writer(self.work_directory)
        self.initialized = True

    @overrides
    def start_profiler(self):
        """
        # see : IMetricTracer.start_profiler()
        """
        if self.is_profiler_enabled:
            tf.profiler.experimental.start(self.work_directory)

    @overrides
    def stop_profiler(self):
        """
        # see : IMetricTracer.stop_profiler()
        """
        if self.is_profiler_enabled:
            tf.profiler.experimental.stop()

    @overrides
    def flush_metrics(self, epoch_number: int) -> None:
        """
        # see : IMetricTracer.flush_metrics(epoch_number)
        """
        with self.summary_writer.as_default():
            for key, value in self.metrics.items():
                if MetricType.SCALAR == key.metric_type:
                    if key.metric_clazz is None:
                        tf.summary.scalar(key.label, data=value, step=epoch_number)
                    else:
                        tf.summary.scalar(key.label, value.result(), step=epoch_number)
                elif MetricType.IMAGE == key.metric_type:
                    tf.summary.image(key.label, value, step=epoch_number)
                elif MetricType.HYPER_PARAMETER == key.metric_type:
                    raise NotImplemented
                elif MetricType.FAIRNESS_INDICATOR == key.metric_type:
                    raise NotImplemented
                elif MetricType.EMBEDDING == key.metric_type:
                    raise NotImplemented

    @overrides
    def update_metric(self, metadata: TraceableMetric, data: MetricData) -> None:
        """
        # see : IMetricTracer.update_metric(metadata, data)
        """
        if metadata is None:
            raise ValueError("No metadata specified")
        if data is None:
            raise ValueError("No data specified")
        metric = self.metrics.get(metadata)
        if metric is not None and metadata.metric_clazz is not None:
            metric(data)
        elif metric is not None:
            self.metrics[metadata] = data
        else:
            if self._log.isEnabledFor(DEBUG):
                self._log.debug("For metric {} the tracing is not enabled from configuration.".format(metadata))

    # protected member functions

    def _build_metrics(self):
        """
        Builds the metrics dictionary based on the provided metadata.
        """
        return dict(map(lambda m: (m, metrics.build_metric_from_metadata(m, tf.keras.metrics.Metric)),
                        self.enabled_metrics))
