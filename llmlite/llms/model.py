from typing import Optional, List, Dict, Any

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
    __config__ = {}

    @classmethod
    def get_config(cls, key: str) -> Any:
        return cls.__config__.get(key)

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

    def completion(self, messages: List[ChatMessage], **kwargs) -> Optional[str]:
        if self.backend_runtime is None:
            raise Exception("Please implement the completion() additionally.")

        prompt = self.prompt(self.model_name_or_path, messages, **kwargs)
        if prompt is None:
            return None

        generated_text = self.backend_runtime.completion(prompt, **kwargs)
        return generated_text

    def validation(self, messages: List[ChatMessage]) -> bool:
        assert len(messages) > 0, "messages should not be empty"
        assert (
            messages[-1].role == consts.USER_PROMPT
        ), "last message should be user prompt"

        if self.get_config("support_system_prompt"):
            for i, msg in enumerate(messages):
                if msg.role == consts.SYSTEM_PROMPT:
                    assert i == 0, "system prompt should be in the first role"
        else:
            for msg in messages:
                assert msg.role != consts.SYSTEM_PROMPT, "system prompt not supported"

        return True

    @classmethod
    def prompt(
        cls, model_name_or_path: str, messages: List[ChatMessage], **kwargs
    ) -> Optional[str]:
        """
        You should implement the prompt additionally because different model
        has different prompt format.
        """

        pass
