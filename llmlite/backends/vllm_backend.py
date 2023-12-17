from typing import Optional, List

from vllm import LLM as vllm

from llmlite.backends.backend import Backend


class VLLMBackend(Backend):
    def __init__(
        self,
        model_name_or_path: str,
        **kwargs,
    ):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        self._model = vllm(
            model=model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def completion(self, content: List[str]) -> Optional[List[str]]:
        sequences = self._model.generate[content)
        response = []
        for seq in sequences:
            response.append(seq.outputs[0].txt)
        return response   
