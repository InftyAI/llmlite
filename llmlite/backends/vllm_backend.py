from typing import Optional

import torch

from llmlite.backends.backend import Backend


class VLLMBackend(Backend):
    def __init__(
        self,
        model_name_or_path: str,
        task: Optional[str],
        torch_dtype: torch.dtype,
        pretrained_model_name: str,
    ):
        pass
