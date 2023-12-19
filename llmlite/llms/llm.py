from typing import Optional
from llmlite import consts
from llmlite.llms.baichuan import Baichuan
from llmlite.llms.chatglm import ChatGLM
from llmlite.llms.llama import Llama
from llmlite.llms.chatgpt import ChatGPT
from llmlite.utils import util


# TODO: loading the configs automatically or construct a register function.
class LLMStore:
    LLMs = {
        consts.MODEL_TYPE_LLAMA: Llama,
        consts.MODEL_TYPE_GPT: ChatGPT,
        consts.MODEL_TYPE_CHATGLM: ChatGLM,
        consts.MODEL_TYPE_BAICHUAN: Baichuan,
    }


class LLM:
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        backend: Optional[str],
        **kwargs,
    ):
        """
        from_pretrained helps to load the model in advance, we support 3 backends right now:
        - endpoint: with this backend, you have to implement the logics additionally, what you
          required to do is to implement the completion() function.
        - hf: basically, no longer need to implement the model loading logic if the model is
          supported by huggingface pipeline, what you required to do is set the model configs only.
          Or you have to implement the load_with_hf() and completion() concurrently.
        - vllm: no longer need to implement the logic since vllm does everything for you if the
          model is supported by vllm already.
        """

        model_class, _ = get_model_info(model_name_or_path)
        backend = get_backend(model_name_or_path, backend)

        # We can call the API directly, no need to load the model.
        if backend == consts.BACKEND_ENDPOINT:
            return model_class(model_name_or_path, **kwargs)

        if backend == consts.BACKEND_HF:
            return model_class.load_with_hf(model_name_or_path, **kwargs)

        if backend == consts.BACKEND_VLLM:
            return model_class.load_with_vllm(model_name_or_path, **kwargs)

        raise Exception("unsupported backend: %s", backend)


# TODO: add tests
def get_model_info(model_name_or_path: str):
    model_type, version = util.parse_model_name(model_name_or_path)

    model_class = LLMStore.LLMs.get(model_type, None)
    assert model_class is not None, "llm not exists"

    return model_class, version


# TODO: add tests
def get_backend(model_name_or_path: str, backend: Optional[str]) -> str:
    model_class, _ = get_model_info(model_name_or_path)

    if backend is None:
        backend = model_class.get_config("default_backend")
        assert backend is not None, "default backend not configured"

    support_backends = model_class.get_config("backends")  # type: ignore
    assert support_backends is not None and backend in support_backends, (
        "only supported backends: " + support_backends
    )

    return backend
