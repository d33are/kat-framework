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

from kat_framework.config.config_props import TensorBoardConfigurationProperty
from kat_framework.framework import KatherineApplication
from kat_framework.util import logger, fileio, metrics
from kat_api import IMetricTracer, TraceableMetric
from kat_typing import MetricData, DistributionStrategy
from overrides import overrides
from typing import Optional
from logging import Logger


WORKING_DIRECTORY_PREFIX = "metrics"


class DummyTracer(IMetricTracer):
    """
    Dummy tracer implementation for testing.
    """

    # protected members

    _log: Logger = None

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        self._log = logger.get_logger(self.__class__.__name__)
        config_handler = KatherineApplication.get_application_config()
        self.enabled_metrics = config_handler.get_config_property(
            TensorBoardConfigurationProperty.ENABLED_NETWORK_METRICS,
            TensorBoardConfigurationProperty.ENABLED_NETWORK_METRICS.prop_type)
        self.is_profiler_enabled = config_handler.get_config_property(
            TensorBoardConfigurationProperty.GPU_PROFILER_ENABLED,
            TensorBoardConfigurationProperty.GPU_PROFILER_ENABLED.prop_type)
        self.work_directory = fileio.build_metrics_work_directory(WORKING_DIRECTORY_PREFIX)
        self.initialized = False

    def is_initialized(self) -> bool:
        """
        # see : IMetricTracer.is_initialized()
        """
        return self.initialized

    def init(self, distribution_strategy: Optional[DistributionStrategy] = None):
        """
        # see : IMetricTracer.init(distribution_strategy)
        """
        self.initialized = True

    @overrides
    def start_profiler(self):
        """
        # see : IMetricTracer.start_profiler()
        """
        pass

    @overrides
    def stop_profiler(self):
        """
        # see : IMetricTracer.stop_profiler()
        """
        pass

    @overrides
    def flush_metrics(self, epoch_number: int) -> None:
        """
        # see : IMetricTracer.flush_metrics(epoch_number)
        """
        pass

    @overrides
    def update_metric(self, metadata: TraceableMetric, data: MetricData) -> None:
        """
        # see : IMetricTracer.update_metric(metadata, data)
        """
        pass

    # protected member functions

    def _build_metrics(self):
        """
        Building dummy test metrics.
        """
        return dict(map(lambda m: (m, metrics.build_metric_from_metadata(m, object)),
                        self.enabled_metrics))
