from typing import Optional

import torch
from vllm import LLM as VLLM 

from llmlite.backends.backend import Backend


class VLLMBackend(Backend):
    def __init__(
        self,
        model_name_or_path: str,
        task: Optional[str],
        torch_dtype: torch.dtype,
        pretrained_model_name: str,
    ):
        self._vllm = VLLM(model=model_name_or_path, trust_remote_code=True)

    def completion(self, content: str) -> Optional[str]:
        sequences = self._vllm.generate([content])
        
        return sequences[0].outputs[0].text
