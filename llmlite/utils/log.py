import logging

from llmlite.utils.envs import LOG_LEVEL


def logging_level():
    return logging.INFO if LOG_LEVEL is None else LOG_LEVEL


logger = logging.getLogger("llmlite")
logger.setLevel(logging_level())
