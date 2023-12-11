from abc import ABC


# TODO: loading the config.json of model file.
class ModelConfig(ABC):
    """
    __config__ =
    {
        "support_system_prompt": True,
        "backends": [consts.BACKEND_HF, consts.BACKEND_VLLM],
        "default_backend": consts.BACKEND_HF,
        "architecture": "LlamaForCausalLM",
    }
    """

    pass
