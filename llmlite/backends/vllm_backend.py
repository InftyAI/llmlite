from typing import Optional, Union, List

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

    def completion(self, content: Union[str, List[str]]) -> Optional[Union[str, List[str]]]:
        sequences = self._model.generate(content)
        if len(sequences) == 1:
            return sequences[0].outputs[0].text
        response = []
        for i in sequences:
            response.append(i.outputs[0].text)
        return response
