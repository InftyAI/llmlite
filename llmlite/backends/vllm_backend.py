from typing import Optional

from vllm import LLM as vllm

from llmlite.backends.backend import Backend


class VLLMBackend(Backend):
    def __init__(
        self,
        model_name_or_path: str,
        architecture: str,
        **kwargs,
    ):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        self._model = vllm(
            model=model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def completion(self, content: str) -> Optional[str]:
        sequences = self._model.generate([content])
        return sequences[0].outputs[0].text
