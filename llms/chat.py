from abc import ABC, abstractmethod
from typing import List

import torch

from apis.messages import ChatMessage

SYSTEM_PROMPT = "system"
USER_PROMPT = "user"
ASSISTANT_PROMPT = "assistant"


# TODO: Support in-memory history in the future, then no need to pass-in the history parameter.
class Chat(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        task: str,
        torch_dtype: torch.dtype,
    ):
        pass

    @abstractmethod
    def validation(self, messages: List[ChatMessage]) -> bool:
        """
        validation helps to validate whether the messages is valid or not, e.g. if LLM supports system_prompt,
        or whether the environment is already, e.g. lack of necessary api keys.

        Args:
            messages: conversations in list.

        Returns:
            A boolean value indicates whether the messages is valid or not.
        """
        pass

    @abstractmethod
    def support_system_prompt(self) -> bool:
        """
        Return:
            A boolean indicates whether support system prompt or not.
        """
        pass

    # TODO: Support history conversation in the future.
    @classmethod
    @abstractmethod
    def prompt(self, messages: List[ChatMessage]) -> str | None:
        pass

    @abstractmethod
    def completion(self, messages: List[ChatMessage]) -> str | None:
        """
        Args:
            system_prompt (str): Not all language models support system prompt, e.g. ChatGLM2.
            user_prompt (str):
        """
        pass


class RemoteChat(Chat):
    """
    RemoteChat is for talking with hosted inference APIs, e.g. ChatGPT, HuggingFace inference APIs.
    """

    pass


class LocalChat(Chat):
    """
    LocalChat is for talking with local models.
    """

    pass
