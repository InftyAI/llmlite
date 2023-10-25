import os
from typing import List
import logging

import torch
import openai  # type: ignore

from llmlite.llms.chat import RemoteChat
from llmlite.llms.messages import ChatMessage


class ChatGPTChat(RemoteChat):
    def __init__(
        self,
        model_name_or_path: str,
        task: str | None,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        """
        This is calling chatgpt api.
        """

        super().__init__(model_name_or_path, task, torch_dtype, **kwargs)

        self.model = model_name_or_path
        self.logger = logging.getLogger("llmlite.ChatGPTChat")

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.endpoint = os.getenv("OPENAI_ENDPOINT")

        openai.api_key = self.api_key
        if self.endpoint is not None:
            openai.api_base = self.endpoint

    @classmethod
    def validate(cls) -> bool:
        key = os.getenv("OPENAI_API_KEY")
        if key is None or key == "":
            logger = logging.getLogger("llmlite.ChatGPTChat")
            logger.error("Error: No OPENAI_API_KEY provided")
            return False
        return True

    @classmethod
    def support_system_prompt(cls) -> bool:
        return True

    def completion(
        self,
        messages: List[ChatMessage],
        **kwargs,
    ) -> str | None:
        inputs = []
        for message in messages:
            inputs.append({"role": message.role, "content": message.content})

        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=inputs,
            **kwargs,
        )

        return completion.choices[0].message.content
