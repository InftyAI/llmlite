from typing import List, no_type_check


import torch

from apis.messages import ChatMessage
from llms.chat import Chat
from llms.chatgpt import ChatGPTChat
from llms.llama import LlamaChat
from llms.chatglm import ChatGLMChat
from utils.log import LOGGER
from apis.utils import general_validations


class ChatLLM:
    """

    How To Use:
        chat = ChatLLM(
            model_name_or_path="<model_name_or_path>",
            task="<task name>", # optional, default to `text-generation`.
            )
        result = chat.completion(
            prompt="<user prompt>",
            system_prompt="<optional system prompt>",
        )

    """

    def __init__(
        self,
        model_name_or_path: str,
        task: str = "text-generation",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            model_name_or_path (str): The model name or the model path.
            task (str): The task defining which pipeline will be returned, default to `text-generation`.
        """

        if model_name_or_path == "":
            raise Exception("model_name_or_path must exist")

        llm = fetch_llm(model_name_or_path)
        self.chat = llm(
            model_name_or_path=model_name_or_path, task=task, torch_dtype=torch_dtype
        )

    def completion(self, messages: List[ChatMessage]) -> str | None:
        """
        Args:
            messages: A list of conversations looks like below:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
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
            LOGGER.warning("general validation not passed")
            return None

        if not self.chat.validation(messages):
            LOGGER.warning("model validation not passed")
            return None

        res = self.chat.completion(messages)
        LOGGER.debug(f"Result: {res}")
        return res


@no_type_check
def fetch_llm(model_name: str) -> Chat:
    model_name = model_name.lower()

    if "llama-2" in model_name:
        return LlamaChat

    if "chatglm2" in model_name:
        return ChatGLMChat

    if "gpt-3.5-turbo" in model_name:
        return ChatGPTChat

    raise UnavailableModelException(
        "model unavailable, supporting model family: `llama-2`, `chatglm2`, `chatgpt`,"
    )


class UnavailableModelException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
