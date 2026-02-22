"""Abstract base class for all defense layers."""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any


class LayerBase(ABC):
    """Abstract base for defense layers. Each layer processes a prompt and returns (action, prompt_or_none, meta)."""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def process(self, prompt: str, context: Dict[str, Any] = None) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """
        Process a prompt.

        Returns:
            (action, transformed_prompt_or_none, meta)
            action: "BLOCK" | "FLAG" | "TRANSFORM" | "ALLOW"
            transformed_prompt_or_none: modified prompt if TRANSFORM, else None
            meta: layer-specific metadata
        """

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"
