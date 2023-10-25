from typing import List
import logging

import torch
import transformers  # type: ignore
from transformers import AutoTokenizer

from llmlite.llms.chat import LocalChat
from llmlite.llms.messages import ChatMessage
from llmlite.utils.validation import general_validations
from llmlite.llms.llama2 import get_full_prompts
from llmlite.llms.chat import SYSTEM_PROMPT


class CodeLlamaChat(LocalChat):
    def __init__(
        self,
        model_name_or_path: str,
        task: str | None,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> None:
        super().__init__(model_name_or_path, task, torch_dtype, **kwargs)

        if task is None:
            task = "text-generation"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

        self.pipeline = transformers.pipeline(
            task,
            model=model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            **kwargs,
        )

        self.logger = logging.getLogger("llmlite.CodeLlamaChat")

    @classmethod
    def validate(cls) -> bool:
        return True

    @classmethod
    def support_system_prompt(cls) -> bool:
        return True

    @classmethod
    def prompt(cls, messages: List[ChatMessage], **kwargs) -> str | None:
        """
        We have `promptMode` in kwargs to choose which prompt we're using, the potential values are:
            - generate, default model.
            - instruct (instruct fine-trained model)
        """

        if not general_validations(messages, cls.support_system_prompt()):
            return None

        logger = logging.getLogger("llmlite.CodeLlamaChat")

        # TODO: support code filling if needed.
        # if mode == "filling":
        #     return f"<PRE> {messages[0].content} <SUF>{messages[1].content} <MID>"

        mode = kwargs.get("promptMode")
        if mode == "generate" and len(messages) > 0:
            logger.warning(
                "We dont' support more than one message in generate mode, we'll use the first message instead"
            )
            if messages[0].role == SYSTEM_PROMPT:
                logger.error("We dont' support system prompt in generate mode")
                return None

            return messages[0].content

        if mode == "instruct":
            return get_full_prompts(messages)

        logger.error("unavailable type: %s", kwargs["promptMode"])
        return None

    def completion(
        self,
        messages: List[ChatMessage],
        **kwargs,
    ) -> str | None:
        mode = kwargs.pop("promptMode", "generate")

        prompt = self.prompt(messages, promptMode=mode)
        if prompt is None:
            return None

        self.logger.debug(f"CodeLlama prompt: {prompt}")

        sequences = self.pipeline(
            prompt,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        return sequences[0]["generated_text"]
