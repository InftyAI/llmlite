from typing import List
import logging

from transformers import AutoTokenizer, AutoModel  # type: ignore
import torch

from llmlite.llms.chat import ASSISTANT_PROMPT, USER_PROMPT, Chat
from llmlite.llms.messages import ChatMessage


class ChatGLMChat(Chat):
    """
    ChatGLM is mainly used for chinese questions and answers. Currently don't support system prompt yet.
    """

    def __init__(
        self,
        model_name_or_path: str,
        task: str = "text-generation",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.model = (
            AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            .half()
            .cuda()
            .eval()
        )
        self.logger = logging.getLogger("llmlite.ChatGLMChat")

    def validation(self, messages: List[ChatMessage]) -> bool:
        return True

    @classmethod
    def support_system_prompt(cls) -> bool:
        return False

    @classmethod
    def prompt(cls, messages: List[ChatMessage]) -> str | None:
        prompt = []

        for message in messages:
            role = message.role
            content = message.content

            if role == USER_PROMPT:
                prompt.append("问：" + content + "\n")
            elif role == ASSISTANT_PROMPT:
                prompt.append("答：" + content + "\n")

        return ("").join(prompt)

    def completion(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_length: int,
        do_sample: bool,
        top_p: float,
        top_k: int,
    ) -> str | None:
        prompt = self.prompt(messages)
        self.logger.debug(f"ChatGLM prompt: {prompt}")

        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            temperature=temperature,
            max_length=max_length,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
        )
        return response
