import logging
import os

LOGGER = logging


def logging_level():
    level = os.getenv("LOGGING_LEVEL")
    return logging.INFO if level is None else level


LOGGER.basicConfig(level=logging_level())
