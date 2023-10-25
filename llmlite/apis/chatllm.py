from typing import List, no_type_check
import logging


import torch

from llmlite.utils.validation import general_validations
from llmlite.llms.chat import Chat
from llmlite.llms.chatgpt import ChatGPTChat
from llmlite.llms.llama2 import LlamaChat
from llmlite.llms.chatglm2 import ChatGLMChat
from llmlite.llms.codellama import CodeLlamaChat
from llmlite.llms.messages import ChatMessage


class ChatLLM:
    """

    How To Use:
        chat = ChatLLM(
            model_name_or_path="meta-llama/Llama-2-7b-chat-hf", # required
            task="text-generation", # optional, default to 'text-generation'
            )

        result = chat.completion(
            messages=[
                ChatMessage(role="system", content="You're a honest assistant."),
                ChatMessage(role="user", content="There's a llama in my garden, what should I do?"),
            ]
        )

    """

    def __init__(
        self,
        model_name_or_path: str,
        task: str | None = None,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        """
        Args:
            model_name_or_path (str): The model name or the model path.
            task (str): The task defining which pipeline will be returned, default to `text-generation`.
        """

        if model_name_or_path == "":
            raise Exception("Error: model_name_or_path must exist")

        llm = fetch_llm(model_name_or_path)
        self.chat = llm(
            model_name_or_path=model_name_or_path,
            task=task,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        self.logger = logging.getLogger("llmlite.ChatLLM")

    def completion(
        self,
        messages: List[ChatMessage],
        **kwargs,
    ) -> str | None:
        """
        Args:
            messages: A list of conversations looks like below:
                [
                    ChatMessage(role="system", content="You are a helpful assistant."),
                    ChatMessage(role="user", content="Who won the world series in 2020?"),
                    ChatMessage(role="assistant", content="The Los Angeles Dodgers won the World Series in 2020."),
                    ChatMessage(role="user", content="Where was it played?"),
                ]

            We have three types of `roles` here:
                system: a system level instruction to guide your model's behavior throughout the conversation.
                user: a series of questions you ask the model.
                assistant: an answer replying to the user questions.
            The `content` contains the text of the message from the role.

            Note: The indexes of the messages are following the sequence of conversations.

        Returns:
            Sentences of string type.
        """

        if not general_validations(messages, self.chat.support_system_prompt()):
            self.logger.warning("general validation not passed")
            return None

        res = self.chat.completion(
            messages=messages,
            **kwargs,
        )
        self.logger.debug(f"Result: {res}")
        return res


@no_type_check
def fetch_llm(model_name: str) -> Chat:
    model_name = model_name.lower()

    if "llama-2" in model_name:
        return LlamaChat

    if "chatglm2" in model_name:
        return ChatGLMChat

    if "gpt-3.5-turbo" in model_name or "gpt-4" in model_name:
        return ChatGPTChat

    if "codellama" in model_name:
        return CodeLlamaChat

    raise UnavailableModelException(
        "model unavailable, supporting model family: `llama-2`, `codellama`, `chatglm2`, `gpt-3.5-turbo`,, `gpt-4`"
    )


class UnavailableModelException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
