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

from kat_api import IObservation
from overrides import overrides
import numpy as np


class DummyObservation(IObservation):
    """
    Dummy observation implementation for testing purposes.
    """

    # public member functions

    def __init__(self, game_state: np.ndarray):
        """
        Default constructor.

        :param game_state:
            current raw game state object
        """
        if game_state is None:
            raise ValueError("No state specified.")
        self._observation_id = 0
        self.screen_buffer = game_state

    def __eq__(self, other):
        if isinstance(other, DummyObservation):
            return self.get_observation_id() == other.get_observation_id()
        return False

    @overrides
    def get_observation_id(self) -> int:
        """
        # see : IObservation.get_observation_id()
        """
        return self._observation_id

    def __str__(self):
        return "Observation number: " + str(self.get_observation_id())
