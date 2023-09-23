from typing import List
import logging

import torch
import openai

from llmlite.llms.chat import RemoteChat
from llmlite.llms.messages import ChatMessage
from llmlite.utils.envs import OPEN_API_KEY


class ChatGPTChat(RemoteChat):
    def __init__(
        self,
        model_name_or_path: str,
        task: str,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        This is calling chatgpt api.
        """

        super().__init__(model_name_or_path, task, torch_dtype)

        self.model = model_name_or_path
        self.logger = logging.getLogger("llmlite.ChatGPTChat")
        openai.api_key = OPEN_API_KEY

    @classmethod
    def validate(cls) -> bool:
        if OPEN_API_KEY is None or OPEN_API_KEY == "":
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
        temperature: float,
        max_length: int,
        do_sample: bool,
        top_p: float,
        top_k: int,
    ) -> str | None:
        logger = logging.getLogger("llmlite.ChatGPTChat")

        if do_sample:
            logger.warning("`do_sample` not support")
            if top_k > 0:
                logger.warning("`top_k` not support")

        inputs = []
        for message in messages:
            inputs.append({"role": message.role, "content": message.content})

        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=inputs,
            temperature=temperature,
            max_tokens=max_length,
            top_p=top_p,
        )

        return completion.choices[0].message
