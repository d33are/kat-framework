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

from kat_framework.framework import KatherineApplication
from kat_framework.monitor.properties import KatMetrics
from kat_framework.drivers.state import KatState
from kat_framework.config.config_props import DriverConfigurationProperty, KatConfigurationProperty
from kat_framework.util import logger
from kat_api import IDriver, IGame, IAgent, StateType, IMetricTracer, IConfigurationHandler
from kat_typing import TrainLoss, MetricData
from abc import abstractmethod
from overrides import overrides
from typing import Optional
from time import sleep
from logging import Logger
import asyncio
import threading


class EpisodeDriver(IDriver):
    """
    Episodic abstract driver implementation.

    This driver builds an IGame, an IAgent and update their states in each tick, based on
    episodic reinforcement learning pattern.
    """

    # protected members

    _log: Logger = None
    _config_handler: IConfigurationHandler = None
    _game: IGame = None
    _agent: IAgent = None
    _metrics: IMetricTracer = None
    _max_episodes: int = None
    _max_steps: int = None
    _sleep_time: int = None
    _action_frequency: int = None
    _training_mode: int = None

    # public member functions

    def __init__(self):
        """
        Default constructor.
        """
        self._log = logger.get_logger(self.__class__.__name__)
        self._config_handler = KatherineApplication.get_application_config()
        if not self._config_handler:
            raise ValueError("No config specified.")
        self._game = KatherineApplication.get_application_factory().build_game()
        self._agent = KatherineApplication.get_application_factory().build_agent()
        self._metrics = KatherineApplication.get_application_factory().build_metrics_tracer()
        self._load_configuration()

    @overrides
    def run(self) -> None:
        """
        Main cycle implementation.

        # see : IDriver.run()
        """
        self._initialize()
        self._loop()
        self._terminate()

    # protected member functions

    @abstractmethod
    def _perform_train_step(self, current_episode: int, current_step: int) -> None:
        """
        Performs a train step. Subsequent implementations can be sync or async train
        functions.

        :param current_episode:
            current episode number
        :param current_step:
            current step in the episode
        """
        pass

    def _initialize(self) -> None:
        """
        Initializing main loop.
        """
        if not self._game.is_initialized():
            self._game.init()
        if not self._agent.is_initialized():
            self._agent.init(self._game.get_observation_space_desc(),
                             self._game.get_action_space_desc())
        if not self._metrics.is_initialized():
            self._metrics.init(self._agent.get_distribution_strategy())
        self._metrics.start_profiler()

    def _loop(self) -> None:
        """
        Main loop.
        """
        global_steps = 0
        for i in range(self._max_episodes):
            is_finished = False
            step_counter = 0
            current_state = KatState(i + 1, self._game.reset(), StateType.INITIAL_STATE)
            while not is_finished:
                action = self._agent.take_action(current_state)
                current_state.set_transition(action)
                next_observation, reward = self._game.make_action(action)
                current_state.set_transitioned_observation(next_observation)
                current_state.set_reward(reward)
                if self._game.is_episode_finished() or self._max_steps <= step_counter:
                    current_state.state_type = StateType.END_STATE
                    self._agent.store_transition(current_state)
                    is_finished = True
                else:
                    self._agent.store_transition(current_state)
                if self._action_frequency > 0:
                    self._game.process_ticks(self._action_frequency)
                    current_state = KatState(
                        i + 1, self._game.get_current_observation(), StateType.ACTIVE_STATE)
                else:
                    current_state = KatState(
                        i + 1, current_state.get_transitioned_observation(), StateType.ACTIVE_STATE)
                if self._sleep_time > 0:
                    sleep(self._sleep_time)
                if self._training_mode:
                    self._perform_train_step(i, global_steps)
                step_counter += 1
                global_steps += 1
            self._update_metrics(exploration_rate=self._agent.get_exploration_rate())
            self._update_metrics(score=self._game.get_total_score())
            self._metrics.flush_metrics(i + 1)

    def _terminate(self) -> None:
        """
        Terminate loop.
        """
        self._metrics.stop_profiler()
        self._agent.persist_model()

    def _load_configuration(self) -> None:
        """
        Loads configuration.
        """
        self._max_episodes = self._config_handler.get_config_property(
            KatConfigurationProperty.MAX_EPISODES,
            KatConfigurationProperty.MAX_EPISODES.prop_type)
        self._max_steps = self._config_handler.get_config_property(
            KatConfigurationProperty.MAX_STEPS,
            KatConfigurationProperty.MAX_STEPS.prop_type)
        self._sleep_time = self._config_handler.get_config_property(
            DriverConfigurationProperty.SLEEP_TIME,
            DriverConfigurationProperty.SLEEP_TIME.prop_type)
        self._action_frequency = self._config_handler.get_config_property(
            DriverConfigurationProperty.ACTION_FREQUENCY,
            DriverConfigurationProperty.ACTION_FREQUENCY.prop_type)
        self._training_mode = self._config_handler.get_config_property(
            DriverConfigurationProperty.TRAINING_ENABLED,
            DriverConfigurationProperty.TRAINING_ENABLED.prop_type)

    def _update_metrics(self,
                        loss: Optional[TrainLoss] = None,
                        exploration_rate: Optional[MetricData] = None,
                        score: Optional[MetricData] = None):
        """
        Helper function for metrics update.

        :param loss:
            loss value
        :param exploration_rate:
            current exploration rate
        :param score:
            total score
        :return:
        """
        if loss is not None:
            self._metrics.update_metric(KatMetrics.TENSORFLOW_TRAIN_LOSS_MEAN, loss)
        if exploration_rate is not None:
            self._metrics.update_metric(KatMetrics.TENSORFLOW_AGENT_EXPLORATION_RATE, exploration_rate)
        if score is not None:
            self._metrics.update_metric(KatMetrics.TENSORFLOW_AGENT_TOTAL_SCORE, score)


class SyncEpisodeDriver(EpisodeDriver, IDriver):
    """
    Synchronous train step driver implementation.
    """
    def __init__(self):
        """
        Default constructor.
        """
        super(SyncEpisodeDriver, self).__init__()

    def _perform_train_step(self, current_episode: int, current_step: int) -> None:
        """
        Performs a train step in synchronous mode.

        :param current_episode:
            current iteration
        :param current_step:
            current step in the episode
        """
        self._agent.tick(current_episode, current_step)
        loss = self._agent.train()
        if isinstance(loss, list):
            loss = loss.pop(0)
        self._update_metrics(loss=loss)


class AsyncEpisodeDriver(EpisodeDriver, IDriver):
    """
    Asynchronous train step driver implementation.

    After the first `perform_train_step` call, it creates a new working thread
    and training the network independently from the driver's main loop.
    """

    _train_loop: asyncio.AbstractEventLoop = None
    _train_thread: threading.Thread = None
    _train_started: bool = None

    def __init__(self):
        """
        Default constructor.
        """
        super(AsyncEpisodeDriver, self).__init__()
        self._init_train_loop()

    @overrides
    def _perform_train_step(self, current_episode: int, current_step: int) -> None:
        """
        Performs a train step in asynchronous mode.

        Starting the worker thread if it haven't started yet, and it calls the agent's
        `tick` method.

        :param current_episode:
            current iteration
        :param current_step:
            current step in the episode
        """
        if not self._train_started:
            self._train_started = True
            self._train_thread.start()
        self._agent.tick(current_episode, current_step)

    @overrides
    def _terminate(self) -> None:
        """
        Overriding the terminate function to stop the working thread
        properly.
        """
        super(AsyncEpisodeDriver, self)._terminate()
        self._train_started = False
        self._train_thread.join()

    def _init_train_loop(self):
        """
        Initializing the working thread.
        """
        if self._training_mode:
            self._train_thread = threading.Thread(name="training", target=self._train_async)

    def _train_async(self):
        """
        Worker thread implementation.
        """
        while self._train_started:
            loss = self._agent.train()
            if isinstance(loss, list):
                loss = loss.pop(0)
            self._update_metrics(loss=loss)
