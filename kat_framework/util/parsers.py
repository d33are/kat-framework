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


import argparse

#DEFAULT_CONFIG_URI = "file://localhost/scenarios/test"
DEFAULT_CONFIG_URI = "file://localhost/scenarios/zdoom/simpler_basic"
#DEFAULT_CONFIG_URI = "file://localhost/scenarios/zdoom/defend_the_line"
#DEFAULT_CONFIG_URI = "file://localhost/scenarios/openai/ram/lunar_lander"
#DEFAULT_CONFIG_URI = "file://localhost/scenarios/openai/img/atari_space_invaders"
#DEFAULT_CONFIG_URI = "file://localhost/scenarios/openai/ram/atari_space_invaders"
#DEFAULT_CONFIG_URI = "file://localhost/scenarios/openai/img/atari_pong"
#DEFAULT_CONFIG_URI = "file://localhost/scenarios/openai/ram/atari_pong"
#DEFAULT_CONFIG_URI = "file://localhost/scenarios/openai/img/atari_breakout"
#DEFAULT_CONFIG_URI = "file://localhost/scenarios/openai/ram/atari_breakout"
DEFAULT_CONFIG_HANDLER = "kat_framework.config.config_handler.YamlConfigHandler"
DEFAULT_FACTORY = "kat_framework.core.factory.KatFactory"
CONFIG_HELP_MSG = "load configs from a resource. (for example: file://localhost/scenarios/test) "
CONFIG_CLASS_HELP_MSG = "(Optional) configuration handler's class name" \
    "(kat_framework.config.config_handler.YamlConfigHandler) "
FACTORY_CLASS_HELP_MSG = "(Optional) application factory class name (kat_framework.core.factory.KatFactory) "


def create_cli_parser(description: str) -> argparse.ArgumentParser:
    """
    Create a parser for the CLI interface.

    :param description:
        Software short description.
    :return:
        the parser object
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_uri", "-cu",
                        dest="config_uri",
                        metavar='CONFIG_URI',
                        type=str,
                        default=DEFAULT_CONFIG_URI,
                        help=CONFIG_HELP_MSG)
    parser.add_argument("--config_class", "-cc",
                        dest="config_class",
                        metavar='CONFIG_CLASS',
                        type=str,
                        default=DEFAULT_CONFIG_HANDLER,
                        help=CONFIG_CLASS_HELP_MSG)
    parser.add_argument("--factory_class", "-fc",
                        dest="factory_class",
                        metavar='FACTORY_CLASS',
                        type=str,
                        default=DEFAULT_FACTORY,
                        help=FACTORY_CLASS_HELP_MSG)
    return parser


def parse_cli_args() -> argparse.Namespace:
    """
    Create a parser instance, and parse the arguments.

    :return:
        parsed arguments in a Namespace
    """
    parser = create_cli_parser(description="Katherine AI for playing video games.")
    return parser.parse_args()
