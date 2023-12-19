import os
from typing import List, Optional, Dict, Any
import logging

import openai  # type: ignore

from llmlite.llms.messages import ChatMessage
from llmlite.llms.model import Model
from llmlite import consts


class ChatGPT(Model):
    def __init__(self, model_name_or_path: str, **kwargs: Dict[str, Any]) -> None:
        super().__init__(model_name_or_path, consts.BACKEND_ENDPOINT, **kwargs)

        self._logger = logging.getLogger("llmlite.ChatGPTChat")

        self._api_key = os.getenv("OPENAI_API_KEY")
        self._endpoint = os.getenv("OPENAI_ENDPOINT")

        if self._api_key is None or self._api_key == "":
            raise Exception("no OPENAI_API_KEY provided")

        openai.api_key = self._api_key
        if self._endpoint is not None:
            openai.api_base = self._endpoint

    __config__ = {
        "support_system_prompt": True,
        "backends": [consts.BACKEND_ENDPOINT],
        "default_backend": consts.BACKEND_ENDPOINT,
    }

    def completion(
        self,
        messages: List[ChatMessage],
        **kwargs,
    ) -> Optional[str]:
        inputs = []
        for message in messages:
            inputs.append({"role": message.role, "content": message.content})

        completion = openai.ChatCompletion.create(
            model=self._model_name_or_path,
            messages=inputs,
            **kwargs,
        )

        return completion.choices[0].message.content
