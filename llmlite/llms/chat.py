from abc import ABC, abstractmethod
from typing import List

import torch

from llmlite.llms.messages import ChatMessage

SYSTEM_PROMPT = "system"
USER_PROMPT = "user"
ASSISTANT_PROMPT = "assistant"


# TODO: Support in-memory history in the future, then no need to pass-in the history parameter.
class Chat(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        task: str | None,
        torch_dtype: torch.dtype,
        **kwargs,
    ) -> None:
        if not self.validate():
            raise Exception("Validate error")

    @classmethod
    @abstractmethod
    def validate(cls) -> bool:
        """
        validation helps to validate whether the environment is already, e.g. lack of necessary api keys.

        Returns:
            A boolean value indicates whether the messages is valid or not.
        """
        pass

    @classmethod
    @abstractmethod
    def support_system_prompt(cls) -> bool:
        """
        Return:
            A boolean indicates whether support system prompt or not.
        """
        pass

    @abstractmethod
    def completion(
        self,
        messages: List[ChatMessage],
        **kwargs,
    ) -> str | None:
        """
        Args:
            messages: a list of chat messages.
            temperature: a parameter that controls the “creativity” or randomness of the text generated.
            max_length: the maximum token size.
            do_sample: bool if set to False greedy decoding is used. Otherwise sampling is used.
            top_p: an alternative to temperature sampling, considering a subset of tokens (the nucleus) whose cumulative probability mass adds up to a certain threshold (top_p).
            top_k: a number of tokens from the highest ranking scores to be considered. This is not required for all LLMs, so optional.
        """
        pass


class RemoteChat(Chat):
    """
    RemoteChat is for talking with hosted inference APIs, e.g. ChatGPT, HuggingFace inference APIs.
    """


class LocalChat(Chat):
    """
    LocalChat is for talking with local models.
    """

    @classmethod
    @abstractmethod
    def prompt(cls, messages: List[ChatMessage], **kwargs) -> str | None:
        pass
