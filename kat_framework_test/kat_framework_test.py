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


from kat_framework import KatherineApplication
import unittest


FACTORY_CLASS = "kat_framework.core.factory.KatFactory"
CONFIG_CLASS = "kat_framework.config.config_handler.YamlConfigHandler"
CONFIG_URI = "file://localhost/scenarios/test"
ENABLE_LOGO = True


class FrameworkTest(unittest.TestCase):
    """
    Runs the test scenario with dummy implementations.
    """
    @staticmethod
    def test_game_loop():
        KatherineApplication.run(FACTORY_CLASS, CONFIG_CLASS, CONFIG_URI, ENABLE_LOGO)


if __name__ == "__main__":
    """
    Application entry point.

    Runs the test scenario with dummy implementations.
    """
    unittest.main()
