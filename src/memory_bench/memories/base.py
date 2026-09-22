"""Contract for ingestion, retrieval, and parametric memory preparation."""

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any
from pathlib import Path

from memory_bench.generators.base import BaseGenerator
from memory_bench.types import MemoryAugmentation, MemoryRecord, Request


class BaseMemory(ABC):
    generation: BaseGenerator | None = None
    generation_settings: dict[str, Any] | None = None

    def bind_generation(self, generator: BaseGenerator, settings: Mapping[str, Any]) -> None:
        """Borrow a runner-owned generator with independent inference settings."""
        self.generation = generator
        self.generation_settings = copy.deepcopy(dict(settings))

    @abstractmethod
    def prepare(self, records: Sequence[MemoryRecord], workspace: Path) -> None:
        """Ingest, index, or train once in the isolated episode workspace."""

    @abstractmethod
    def augment(self, request: Request) -> MemoryAugmentation:
        """Return context and/or an adapter without learning from predictions."""

    @abstractmethod
    def close(self) -> None:
        """Release resources, including after partially completed preparation."""
