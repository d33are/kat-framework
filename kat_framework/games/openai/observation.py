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

from kat_api import IObservation, NetworkInputType
from overrides import overrides
import numpy as np


class OpenAIObservation(IObservation):
    """
    OpenAI based games observation implementation.

    Mostly openai games have a Box (ArrayLike) observation.
    """

    # public member functions

    def __init__(self, game_state: np.ndarray, network_input: NetworkInputType):
        """
        Default constructor.

        :param game_state:
            current game state object
        """
        if game_state is None:
            raise ValueError("No state specified.")
        self._observation_id = 0
        if NetworkInputType.RAM == network_input:
            self.ram_vector = game_state
        elif NetworkInputType.IMG == network_input:
            self.screen_buffer = game_state
        else:
            raise ValueError("{} input type is not supported.".format(str(network_input)))

    def __eq__(self, other):
        if isinstance(other, OpenAIObservation):
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
