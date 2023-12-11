from typing import List

from llmlite import consts
from llmlite.llms.messages import ChatMessage
from llmlite.utils.log import logger


def general_validations(
    messages: List[ChatMessage], support_system_prompt: bool
) -> bool:
    if len(messages) == 0:
        logger.error("no message provided")
        return False

    if not support_system_prompt:
        for message in messages:
            if message.role == consts.SYSTEM_PROMPT:
                logger.error("system prompt not supported")
                return False

    return True
