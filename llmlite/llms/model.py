from typing import Optional, List, Dict, Any

import torch

from llmlite.llms.messages import ChatMessage
from llmlite.backends.hf_backend import HFBackend
from llmlite.backends.vllm_backend import VLLMBackend


class Model:
    def __init__(
        self,
        model_name_or_path: str,
        task: Optional[str],
        torch_dtype: torch.dtype,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._task = task
        self._torch_dtype = torch_dtype

    # Each model should have its own configuration.
    __config__ = Dict[str, Any]

    @classmethod
    def get_config(cls, key: str) -> Any:
        return cls.__config__.get(key, None)  # type: ignore

    # You should implement this if the model has a different architecture or has not supported
    # by huggingface pipeline yet. Also, you should implement the completion() the same time.
    def load_with_hf(
        self,
        architecture: str,
    ):
        self._backend = HFBackend(
            self._model_name_or_path,
            self._task,
            self._torch_dtype,
            architecture,
        )

    # You should implement this if the model has a different architecture or has not supported
    # by huggingface pipeline yet. Also, you should implement the completion() the same time.
    def load_with_vllm(
        self,
        architecture: str,
    ):
        self._hf_backend = False
        self._backend = VLLMBackend(
            self._model_name_or_path,
            self._task,
            self._torch_dtype,
            architecture,
        )

    def completion(self, messages: List[ChatMessage]) -> Optional[str]:
        if self._backend is None:
            raise Exception("Please implement the completion() additionally.")

        prompt = self.prompt(messages)
        if prompt is None:
            return None

        generated_text = self._backend.completion(prompt)
        return generated_text

    @classmethod
    def prompt(cls, messages: List[ChatMessage]) -> Optional[str]:
        """
        You should implement the prompt additionally because different model
        has different prompt format.
        """

        raise Exception("not implemented")
