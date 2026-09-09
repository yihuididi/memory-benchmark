"""Contract for the common answering pipeline used across memory variants."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from memory_bench.generators.base import BaseGenerator
from memory_bench.memories.base import BaseMemory
from memory_bench.types import Prediction, Request


class BaseAgent(ABC):
    @abstractmethod
    def answer(
        self,
        request: Request,
        memory: BaseMemory,
        generation: BaseGenerator,
        settings: Mapping[str, Any],
    ) -> Prediction:
        """Answer independently without receiving evaluation references or IDs."""
