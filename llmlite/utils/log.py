import logging
import os


def logging_level():
    level = os.getenv("LOG_LEVEL")
    return logging.INFO if level is None else level


logger = logging.getLogger("llmlite")
logger.setLevel(logging_level())
