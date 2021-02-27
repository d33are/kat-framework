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

import importlib
from typing import Optional
from kat_framework.util import logger
from typing import TypeVar, Type

ATTRIBUTE_ERROR_MSG = "Class doesn't exists: %s"
SUBCLASS_ERROR_MSG = "Not a(n) %s implementation."

log = logger.get_logger(__name__)

T = TypeVar('T')


def get_class_from_name(class_canonical_name: str, check_is_subclass: Optional[Type[T]] = None) -> T:
    """
    Resolve the class from the specified string.

    Optional: If the caller pass the 'check_is_subclass' variable, the function calls
    an 'issubclass(resolved_class, check_is_subclass)' for inheritance checking.

    :param check_is_subclass:
        any object, or None, the default is None
    :param class_canonical_name:
        the canonical name (string) of the class (for ex: package_name.module_name.class_name)
    :return:
        the resolved class
    :raises ModuleNotFoundError
        if can't import the specified module
            AttributeError
        if the specified class cannot be found in the module
    """
    module_name, class_name = class_canonical_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    clazz = None
    try:
        clazz = getattr(module, class_name)
    except AttributeError:
        log.critical(ATTRIBUTE_ERROR_MSG, class_canonical_name)
    if check_is_subclass and not issubclass(clazz, check_is_subclass):
        log.critical(SUBCLASS_ERROR_MSG, str(check_is_subclass))
    return clazz


def get_instance(class_canonical_name: str, check_is_subclass: Optional[Type[T]] = None,
                 *constructor_args: Optional[object]) -> T:
    """
    Resolve the class from the specified string, and instantiate it.

    Optional: If the caller pass the 'check_is_subclass' variable, the function calls
    an 'issubclass(resolved_class, check_is_subclass)' for inheritance checking.

    :param class_canonical_name:
        the canonical name (string) of the class (for ex: package_name.module_name.class_name)
    :param check_is_subclass:
        any object, or None, the default is None
    :param constructor_args:
        optional constructor parameters for the class instantiation
    :return:
        an instance of the resolved class
    """
    if not class_canonical_name:
        raise ValueError("No class_canonical_name specified.")
    resolved_class = get_class_from_name(class_canonical_name, check_is_subclass)
    return resolved_class(*constructor_args)
