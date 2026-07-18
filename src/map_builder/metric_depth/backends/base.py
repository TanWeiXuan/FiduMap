from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MetricDepthBackend(ABC):
    def __init__(self) -> None:
        self.model: Any | None = None
        self.processor: Any | None = None
        self.device = "cpu"

    @abstractmethod
    def load(self, config: Any) -> None: ...

    @abstractmethod
    def infer(self, image_rgb: Any, prompt: Any, camera_model: Any, config: Any, progress: Any = None) -> Any: ...

    def close(self) -> None:
        self.model = None
        self.processor = None
