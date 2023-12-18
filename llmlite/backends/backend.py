from abc import ABC, abstractmethod
from typing import Optional


class Backend(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def completion(self, content: str, **kwargs) -> Optional[str]:
        pass
