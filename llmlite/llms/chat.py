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
    def completion(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_length: int,
        do_sample: bool,
        top_p: float,
        top_k: int,
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

    pass


class LocalChat(Chat):
    """
    LocalChat is for talking with local models.
    """

    pass
