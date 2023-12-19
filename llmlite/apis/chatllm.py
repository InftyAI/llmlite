from typing import List, Optional, Union
import logging

from llmlite import consts
from llmlite.llms.llm import LLM, get_model_info
from llmlite.llms.messages import ChatMessage


class ChatLLM:
    """
    How To Use:
        chat = ChatLLM(
            model_name_or_path="meta-llama/Llama-2-7b-chat-hf", # required
            task="text-generation", # optional
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
        backend: str = None,
        **kwargs,
    ):
        """
        Args:
            model_name_or_path (str): The model name or the model path.
            task (str): The task defining which pipeline will be returned, default to `text-generation`.
        """

        if model_name_or_path == "":
            raise Exception("model_name_or_path must exist")

        if backend not in [
            None,
            consts.BACKEND_ENDPOINT,
            consts.BACKEND_HF,
            consts.BACKEND_VLLM,
        ]:
            raise Exception("backend not support")

        self._llm = LLM.from_pretrained(
            model_name_or_path=model_name_or_path,
            backend=backend,
            **kwargs,
        )

        self.logger = logging.getLogger("llmlite.ChatLLM")

    def completion(
        self,
        messages: Union[List[ChatMessage], List[List[ChatMessage]]],
        **kwargs,
    ) -> Optional[Union[str, List[str]]]:
        """
        Args:
            messages: A list of conversations looks like below:
                [
                    ChatMessage(role="system", content="You are a helpful assistant."),
                    ChatMessage(role="user", content="Who won the world series in 2020?"),
                    ChatMessage(role="assistant", content="The Los Angeles Dodgers won the World Series in 2020."),
                ]

            We have three types of `roles` here:
                system: a system level instruction to guide your model's behavior throughout the conversation.
                user: a series of questions you ask the model.
                assistant: an answer replying to the user questions.
            The `content` contains the text of the message from the role.

            Note: The indexes of the messages are following the sequence of conversations.

        Returns:
            Sentences of string type or a list of string type sentences, this will only happen with vLLM backend.
        """

        self._llm.validation(messages=messages)

        res = self._llm.completion(messages=messages, **kwargs)
        self.logger.debug(f"Result: {res}")
        return res

    @classmethod
    def prompt(
        cls, model_name_or_path: str, messages: List[ChatMessage], **kwargs
    ) -> Optional[str]:
        model_class, _ = get_model_info(model_name_or_path)
        return model_class.prompt(model_name_or_path, messages, **kwargs)
