from typing import List

from llmlite.llms.chat import SYSTEM_PROMPT
from llmlite.llms.messages import ChatMessage
from llmlite.utils.log import logger


def general_validations(
    messages: List[ChatMessage], support_system_prompt: bool
) -> bool:
    if len(messages) == 0:
        logger.error("no prompt provided")
        return False

    if messages[0].role == SYSTEM_PROMPT and not support_system_prompt:
        logger.error("system prompt not supported")
        return False

    return True
