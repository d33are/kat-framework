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


class IObservation(metaclass=ABCMeta):
    """
    Common interface for observations.

    Observation represents a specific state of a game instance.
    This is just a marker interface.
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
        return (hasattr(subclass, 'get_observation_id') and
                callable(subclass.get_observation_id) or
                NotImplemented)

    @abstractmethod
    def get_observation_id(self) -> int:
        """
        Observation id.

        :return:
            the unique id of the observation
        """
        pass
