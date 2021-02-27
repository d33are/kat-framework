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
from vizdoom.vizdoom import GameState


class DoomObservation(IObservation):
    """
    `IObservation implementation for zDoom.

    All game buffer is mapped to a python buffer.
    """

    def __init__(self, game_state: GameState):
        """
        Default constructor.

        :param game_state:
            current game state from zDoom
        """
        if game_state is None:
            raise ValueError("No state specified.")
        self._observation_id = game_state.number
        self.screen_buffer = game_state.screen_buffer
        self.game_variables = game_state.game_variables
        self.depth_buffer = game_state.depth_buffer
        self.label_buffer = game_state.labels_buffer
        self.automap_buffer = game_state.automap_buffer
        self.game_labels = game_state.labels
        self.game_objects = game_state.objects
        self.game_sectors = game_state.sectors

    def __eq__(self, other):
        if isinstance(other, DoomObservation):
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
