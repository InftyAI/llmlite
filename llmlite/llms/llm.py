from typing import Optional

import torch

from llmlite import consts
from llmlite.llms.chatglm import ChatGLM
from llmlite.llms.llama import Llama
from llmlite.llms.chatgpt import ChatGPT


# TODO: loading the configs automatically or construct a register function.
class LLMStore:
    LLMs = {
        consts.ARCH_LLAMA: Llama,
        consts.ARCH_GPT: ChatGPT,
        consts.ARCH_CHATGLM: ChatGLM,
    }


class LLM:
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        task: Optional[str],
        torch_dtype: torch.dtype,
        backend: str,
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
        arch = get_model_arch(model_name_or_path)
        llm_class = LLMStore.LLMs.get(arch, None)
        if llm_class is None:
            raise Exception("llm not exists")

        model = llm_class(model_name_or_path, task, torch_dtype)

        backend = get_backend(backend, arch)
        
        # We can call the API directly, no need to load the model.
        if backend == consts.BACKEND_ENDPOINT:
            return model

        architecture = model.get_config("architecture")
        if architecture is None:
            raise Exception("architecture not exists in config")        

        if backend == consts.BACKEND_HF:
            model.load_with_hf(architecture)
            return model

        if backend == consts.BACKEND_VLLM:
            model.load_with_vllm(architecture)
            return model

        raise Exception("unsupported backend: %s", backend)


def get_backend(backend: str, arch: str) -> str:
    # TODO: this should be set manually, make it automatically.
    if arch in [consts.ARCH_GPT]:
        return consts.BACKEND_ENDPOINT

    llm_class = LLMStore.LLMs[arch]
    if llm_class is None:
        raise Exception("llm not exists")
    support_backends = llm_class.get_config("backends")  # type: ignore

    if backend in support_backends:
        return backend

    # fallback to default backend
    default_backend = llm_class.get_config("default_backend")  # type: ignore
    if default_backend is None:
        raise Exception("no default backend set")

    return default_backend


def get_model_arch(model_name: str) -> str:
    model_name = model_name.lower()

    if "llama" in model_name:
        return consts.ARCH_LLAMA

    if "chatglm" in model_name:
        return consts.ARCH_CHATGLM

    if "gpt-3.5" in model_name or "gpt-4" in model_name:
        return consts.ARCH_GPT

    raise UnavailableModelException(
        "model unavailable, supporting model family: `llama`,, `chatglm`, `chatgpt`"
    )


class UnavailableModelException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
