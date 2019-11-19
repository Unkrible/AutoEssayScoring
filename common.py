# pylint: disable=logging-fstring-interpolation, broad-except
"""common"""
import signal
import math
import logging
import functools
import os
import time
import sys
from contextlib import contextmanager
from typing import Any

from constant import *


def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
        2019-04-25 12:52:51,924 INFO score.py: <message>
    """
    logger = logging.getLogger("AES")
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s: %(message)s'
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


VERBOSITY_LEVEL = DEBUG
_LOGGER = get_logger(VERBOSITY_LEVEL)

nesting_level = 0

_LOGGER_MAP = {
    INFO: _LOGGER.info,
    WARNING: _LOGGER.warning,
    DEBUG: _LOGGER.debug,
    ERROR: _LOGGER.error
}


def log(entry: Any, level=INFO):
    global nesting_level
    space = "-" * (4 * nesting_level)
    _LOGGER_MAP[level](f"{space}{entry}")


def timeit(method):
    @functools.wraps(method)
    def timed(*args, **kw):
        global is_start
        global nesting_level

        class_name = ""
        if len(args) > 0 and hasattr(args[0], '__class__'):
            class_name = f"{args[0].__class__.__name__}."
        is_start = True
        log(f"Start [{class_name}{method.__name__}]:")
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{class_name}{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        is_start = False

        return result

    return timed


def _here(*args):
    """Helper function for getting the current directory of this script."""
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(here, *args))
