import torch
from typing import Optional

import transformers  # type: ignore
from transformers import AutoTokenizer

from llmlite.backends.backend import Backend
from llmlite.utils.util import get_class


class HFBackend(Backend):
    def __init__(
        self,
        model_name_or_path: str,
        task: Optional[str],
        torch_dtype: torch.dtype,
        architecture: str,
    ):
        model_class = get_class("transformers", architecture)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        model = model_class.from_pretrained(model_name_or_path).half().cuda().eval()

        self._pipeline = transformers.pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            device=0,
        )

    def completion(self, content: str) -> Optional[str]:
        sequences = self._pipeline(
            content,
            return_full_text=False,
        )

        return sequences[0]["generated_text"]
