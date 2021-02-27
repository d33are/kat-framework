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
from kat_api.state import IState


class IDriver(metaclass=ABCMeta):
    """
    Common interface for drivers.

    The purpose of a driver is to handle the interactions between:
     * agents
     * environments
     * monitoring

     We're assuming episodic step driver implementations.
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
        return (hasattr(subclass, 'run') and
                callable(subclass.run) or
                NotImplemented)

    @abstractmethod
    def run(self) -> IState:
        """
        Steps the environment using actions from the agent until at least one of the following
        termination criteria is met:
            The number of steps reaches max_steps or,
            the number of episodes reaches max_episodes.
        :return:
            end state
        """
        pass
