from typing import List, Optional, Dict, Any
import logging

import torch
from transformers import AutoTokenizer, GenerationConfig  # type: ignore

from llmlite.llms.model import Model
from llmlite import consts
from llmlite.llms.messages import ChatMessage
from llmlite.utils import util

user_token = "<reserved_106>"
assistant_token = "<reserved_107>"


class Baichuan(Model):
    def __init__(
        self,
        model_name_or_path: str,
        backend: str,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(model_name_or_path, backend, **kwargs)

    __config__ = {
        "support_system_prompt": True,
        "backends": [consts.BACKEND_HF],
        "default_backend": consts.BACKEND_HF,
        "architecture": "AutoModelForCausalLM",
    }

    @classmethod
    def load_with_hf(
        cls,
        model_name_or_path: str,
        **kwargs: Dict[str, Any],
    ):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        use_fast = kwargs.pop("use_fast", False)
        device_map = kwargs.pop("device_map", "auto")
        torch_type = kwargs.pop("torch_type", torch.bfloat16)

        arch = cls.get_config("architecture")
        if arch is None:
            raise Exception("architecture not exists")

        model_class = util.get_class("transformers", arch)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
        model = model_class.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_type,
            device_map=device_map,
            **kwargs,
        )
        model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

        config_args = {
            "model": model,
            "tokenizer": tokenizer,
            "logger": logging.getLogger("llmlite.Baichuan"),
        }
        return cls(model_name_or_path, consts.BACKEND_HF, **config_args)

    def completion(
        self,
        messages: List[ChatMessage],
        **kwargs,
    ) -> Optional[str]:
        inputs = []
        for msg in messages:
            inputs.append({"role": msg.role, "content": msg.content})
        response = self._model.chat(self._tokenizer, inputs)
        return response

    @classmethod
    def prompt(
        cls,
        model_name_or_path: str,
        messages: List[ChatMessage],
        **kwargs,
    ) -> Optional[str]:
        super().prompt(model_name_or_path, messages, **kwargs)

        # This is inspired by https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/generation_utils.py
        system_prompt = None

        if len(messages) > 0 and messages[0].role == consts.SYSTEM_PROMPT:
            system_prompt = messages[0].content + "\n"
            messages = messages[1 : len(messages)]

        prompts = ""
        for msg in messages:
            if msg.role == consts.USER_PROMPT:
                prompts += user_token + msg.content
            if msg.role == consts.ASSISTANT_PROMPT:
                prompts += assistant_token + msg.content

        if system_prompt is None:
            return prompts + assistant_token
        else:
            return system_prompt + prompts + assistant_token
