from typing import Optional

import transformers  # type: ignore
from transformers import AutoTokenizer, AutoConfig

from llmlite.backends.backend import Backend

# from llmlite.utils import util


class HFBackend(Backend):
    def __init__(
        self,
        model_name_or_path: str,
        architecture: str,
        **kwargs,
    ):
        task = kwargs.pop("task", None)
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        device_map = kwargs.pop("device_map", "auto")

        # model_class = util.get_class("transformers", architecture)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        # TODO: remove this if model path is already supported
        # model = model_class.from_pretrained(model_name_or_path).half().cuda().eval()
        config = AutoConfig.from_pretrained(model_name_or_path)

        self._pipeline = transformers.pipeline(
            task=task,
            model=model_name_or_path,
            tokenizer=tokenizer,
            config=config,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            **kwargs,
        )

    def completion(self, content: str, **kwargs) -> Optional[str]:
        sequences = self._pipeline(
            content,
            return_full_text=False,
            **kwargs,
        )
        return sequences[0]["generated_text"]
