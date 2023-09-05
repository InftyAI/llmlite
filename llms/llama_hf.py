import torch

from llms.chat import APIChat


class LlamaHFChat(APIChat):
    """
    This is for huggingface inference hosted api. See https://huggingface.co/docs/api-inference/index.
    The url looks like https://api-inference.huggingface.co/models/{model}.
    """

    def __init__(
        self,
        model_name_or_path: str = None,
        task: str = None,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        raise Exception("not implemented")
