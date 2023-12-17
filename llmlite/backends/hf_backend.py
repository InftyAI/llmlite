from typing import Optional

import transformers  # type: ignore
from transformers import AutoTokenizer, AutoConfig

from llmlite.backends.backend import Backend

# from llmlite.utils.util import get_class


class HFBackend(Backend):
    def __init__(
        self,
        model_name_or_path: str,
        architecture: str,
        **kwargs,
    ):
        task = kwargs.pop("task", None)
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        torch_dtype = kwargs.pop("torch_dtype", None)

        # model_class = get_class("transformers", architecture)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        # model = model_class.from_pretrained(model_name_or_path).half().cuda().eval()
        config = AutoConfig.from_pretrained(model_name_or_path)

        self._pipeline = transformers.pipeline(
            task=task,
            model=model_name_or_path,
            tokenizer=tokenizer,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            **kwargs,
        )

    def completion(self, content: str) -> Optional[str]:
        sequences = self._pipeline(
            content,
            return_full_text=False,
        )
        return sequences[0]["generated_text"]
