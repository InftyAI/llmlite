from typing import Optional, List, Dict, Any, Union

from llmlite.llms.messages import ChatMessage
from llmlite.backends.hf_backend import HFBackend
from llmlite.backends.vllm_backend import VLLMBackend
from llmlite import consts
from llmlite.utils import util


class Model:
    def __init__(
        self,
        model_name_or_path: str,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.model_name_or_path = model_name_or_path
        _, self.version = util.parse_model_name(model_name_or_path)

        self.__dict__.update(kwargs)

    # Each model should have its own configuration.
    __config__ = Dict[str, Any]

    @classmethod
    def get_config(cls, key: str) -> Any:
        return cls.__config__.get(key, None)

    # You should implement this if the model has a different architecture or has not supported
    # by huggingface pipeline yet. Also, you should implement the completion() the same time.
    @classmethod
    def load_with_hf(
        cls,
        model_name_or_path: str,
        **kwargs: Dict[str, Any],
    ):
        arch = cls.get_config("architecture")
        if arch is None:
            raise Exception("architecture not exists")

        backend_runtime = HFBackend(
            model_name_or_path,
            arch,
            **kwargs,
        )

        config_args = {
            "backend": consts.BACKEND_HF,
            "backend_runtime": backend_runtime,
        }
        return cls(model_name_or_path, **config_args)

    # You should implement this if the model has a different architecture or has not supported
    # by huggingface pipeline yet. Also, you should implement the completion() the same time.
    @classmethod
    def load_with_vllm(
        cls,
        model_name_or_path: str,
        **kwargs: Dict[str, Any],
    ):
        arch = cls.get_config("architecture")
        if arch is None:
            raise Exception("architecture not exists")

        backend_runtime = VLLMBackend(
            model_name_or_path,
            arch,
            **kwargs,
        )

        config_args = {
            "backend": consts.BACKEND_VLLM,
            "backend_runtime": backend_runtime,
        }
        return cls(model_name_or_path, **config_args)

    def completion(self, messages: Union[List[ChatMessage], List[List[ChatMessage]]], **kwargs) -> Optional[Union[str, List[str]]]:
        if self.backend_runtime is None:
            raise Exception("Please implement the completion() additionally.")

        prompt = []
        if type(messages[0]) == list:
            prompt = self.prompt(self.model_name_or_path, messages, **kwargs)
        else:
            for message in messages:
                mes = self.prompt(self.model_name_or_path, messages, **kwargs)
                prompt.append(mes)
        if prompt is None:
            return None

        generated_text = self.backend_runtime.completion(prompt)
        return generated_text

    def validation(self, messages: List[ChatMessage]) -> bool:
        if (
            not self.get_config("support_system_prompt")
            and len(messages) > 0
            and messages[0].role == consts.SYSTEM_PROMPT
        ):
            raise Exception("system prompt not supported yet")

    @classmethod
    def prompt(
        cls, model_name_or_path: str, messages: List[ChatMessage], **kwargs
    ) -> Optional[str]:
        """
        You should implement the prompt additionally because different model
        has different prompt format.
        """

        pass
