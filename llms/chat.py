from abc import ABC, abstractmethod

import torch

SYSTEM_PROMPT = "system"
USER_PROMPT = "user"


# TODO: Support in-memory history in the future, then no need to pass-in the history parameter.
class Chat(ABC):
    def __init__(
        self,
        model_name_or_path: str = None,
        task: str = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        pass

    @classmethod
    @abstractmethod
    def support_system_prompt() -> bool:
        """
        Return:
            A boolean indicates whether support system prompt or not.
        """
        pass

    # TODO: Support history conversation in the future.
    @classmethod
    @abstractmethod
    def prompt(
        self,
        system_prompt: str = None,
        user_prompt: str = None,
    ) -> str:
        pass

    @abstractmethod
    def completion(
        self,
        system_prompt: str = None,
        user_prompt: str = None,
    ) -> str:
        """
        Args:
            system_prompt (str): Not all language models support system prompt, e.g. ChatGLM2.
            user_prompt (str):
        """
        pass


class APIChat(Chat):
    """
    APIChat is for talking with hosting inference APIs, e.g. ChatGPT, HuggingFace inference APIs.
    """

    pass


class LocalChat(Chat):
    """
    LocalChat is for talking with local models.
    """

    pass
