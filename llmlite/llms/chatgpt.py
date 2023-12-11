import os
from typing import List, Optional
import logging

import torch
import openai  # type: ignore

from llmlite.llms.messages import ChatMessage
from llmlite.llms.model import Model


class ChatGPT(Model):
    def __init__(
        self,
        model_name_or_path: str,
        task: Optional[str],
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__(model_name_or_path, task, torch_dtype)

        self.logger = logging.getLogger("llmlite.ChatGPTChat")

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.endpoint = os.getenv("OPENAI_ENDPOINT")

        if self.api_key is None or self.api_key == "":
            raise Exception("no OPENAI_API_KEY provided")

        openai.api_key = self.api_key
        if self.endpoint is not None:
            openai.api_base = self.endpoint

    __config__ = {
        "support_system_prompt": True,
    }

    def completion(
        self,
        messages: List[ChatMessage],
    ) -> Optional[str]:
        inputs = []
        for message in messages:
            inputs.append({"role": message.role, "content": message.content})

        completion = openai.ChatCompletion.create(
            model=self._model_name_or_path,
            messages=inputs,
        )

        return completion.choices[0].message.content
