from typing import Tuple
import importlib

from llmlite import consts


def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# Return the model_type and the version together.
def parse_model_name(model_name: str) -> Tuple[str, float]:
    model_name = model_name.lower()

    if "llama2" in model_name or "llama-2" in model_name or "codellama" in model_name:
        return consts.MODEL_TYPE_LLAMA, 2

    if "chatglm2" in model_name:
        return consts.MODEL_TYPE_CHATGLM, 2

    if "chatglm3" in model_name:
        return consts.MODEL_TYPE_CHATGLM, 3

    if "baichuan2" in model_name:
        return consts.MODEL_TYPE_BAICHUAN, 2

    if "gpt-3.5" in model_name:
        return consts.MODEL_TYPE_GPT, 3.5

    if "gpt-4" in model_name:
        return consts.MODEL_TYPE_GPT, 4

    raise UnavailableModelException(
        "model unavailable, supporting model family: `llama`,, `chatglm`, `chatgpt`, `baichuan`"
    )


class UnavailableModelException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
