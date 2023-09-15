import torch
from llmlite.llms.chat import RemoteChat


class HuggingFaceChat(RemoteChat):
    def __init__(
        self,
        model_name_or_path: str,
        task: str,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        This is huggingface hosted inference api.
        """
        pass
