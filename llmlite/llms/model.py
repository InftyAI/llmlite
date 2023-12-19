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
        backend: str,
        **kwargs: Dict[str, Any],
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._backend = backend
        _, self._version = util.parse_model_name(model_name_or_path)

        updates = {"_" + k: v for k, v in kwargs.items()}
        self.__dict__.update(updates)

    # Each model should have its own configuration.
    __config__ = {}

    @property
    def backend(self) -> str:
        return self._backend

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

        return cls(
            model_name_or_path,
            backend=consts.BACKEND_HF,
            backend_runtime=backend_runtime,
        )

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
            **kwargs,
        )

        return cls(
            model_name_or_path,
            backend=consts.BACKEND_VLLM,
            backend_runtime=backend_runtime,
        )

    def completion(
        self, messages: Union[List[ChatMessage], List[List[ChatMessage]]], **kwargs
    ) -> Optional[Union[str, List[str]]]:
        if self._backend == consts.BACKEND_HF:
            prompt = self.prompt(self._model_name_or_path, messages, **kwargs)

            generated_text = self._backend_runtime.completion(prompt, **kwargs)
            return generated_text

        if self._backend == consts.BACKEND_VLLM:
            prompts = []
            for msg in messages:
                pt = self.prompt(self._model_name_or_path, msg, **kwargs)
                prompts.append(pt)

            generated_text = self._backend_runtime.completion(prompts, **kwargs)
            return generated_text

    def validation(
        self,
        messages: Union[List[ChatMessage], List[List[ChatMessage]]],
    ):
        if self._backend == consts.BACKEND_VLLM:
            assert all(
                isinstance(item, list) for item in messages
            ), "vLLM only supports batch inference"
        else:
            assert all(
                isinstance(item, ChatMessage) for item in messages
            ), "batch inference only supports with vLLM backend"

        assert len(messages) > 0, "messages should not be empty"

        def validate(messages: List[ChatMessage]):
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
                    assert (
                        msg.role != consts.SYSTEM_PROMPT
                    ), "system prompt not supported"

        if all(isinstance(item, list) for item in messages):
            [validate(messages=item) for item in messages]
        else:
            validate(messages=messages)

    @classmethod
    def prompt(
        cls, model_name_or_path: str, messages: List[ChatMessage], **kwargs
    ) -> Optional[str]:
        """
        You should implement the prompt additionally because different model
        has different prompt format.
        """

        pass
