from abc import ABC, abstractmethod
from typing import Optional

import torch


class Backend(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        task: Optional[str],
        torch_dtype: torch.dtype,
        pretrained_model_name: str,
    ) -> None:
        pass

    @abstractmethod
    def completion(self, content: str) -> Optional[str]:
        pass
