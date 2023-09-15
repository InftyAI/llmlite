from typing import List

from transformers import AutoTokenizer, AutoModel  # type: ignore
import torch

from llmlite.llms.chat import ASSISTANT_PROMPT, USER_PROMPT, Chat
from llmlite.apis.messages import ChatMessage
from llmlite.utils.log import LOGGER


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
        temperature: float = 0.2,
        max_length: int = 2048,
        top_p: float = 0.7,
        top_k: int | None = None,  # We do not use top_k in ChatGLM
    ) -> str | None:
        prompt = self.prompt(messages)
        LOGGER.debug(f"ChatGLM prompt: {prompt}")

        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            temperature=temperature,
            max_length=max_length,
            top_p=top_p,
        )
        return response
