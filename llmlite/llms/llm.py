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
        backend: str,
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

        model_class, _, backend = get_model_info(backend, model_name_or_path)

        # We can call the API directly, no need to load the model.
        if backend == consts.BACKEND_ENDPOINT:
            return model_class

        if backend == consts.BACKEND_HF:
            return model_class.load_with_hf(model_name_or_path, **kwargs)

        if backend == consts.BACKEND_VLLM:
            return model_class.load_with_vllm(model_name_or_path, **kwargs)

        raise Exception("unsupported backend: %s", backend)


def get_model_info(backend: str, model_name_or_path: str):
    model_type, version = util.parse_model_name(model_name_or_path)

    model_class = LLMStore.LLMs.get(model_type, None)
    if model_class is None:
        raise Exception("llm not exists")

    support_backends = model_class.get_config("backends")  # type: ignore
    if backend not in support_backends:
        # fallback to default backend
        backend = model_class.get_config("default_backend", None)  # type: ignore
        if backend is None:
            raise Exception("no default backend set")

    return model_class, version, backend
