from abc import ABC, abstractmethod
from typing import Optional, List

import torch


class Backend(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        architecture: str,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def completion(self, content: str) -> Optional[str]:
        pass


class BatchBackend(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        task: Optional[str],
        torch_dtype: torch.dtype,
    ) -> None:
        pass

    @abstractmethod
    def BatchCompletion(self, content: List[str]) -> Optional[str]:
        pass
