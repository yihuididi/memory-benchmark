"""Contract for ingestion, retrieval, and parametric memory preparation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from memory_bench.types import MemoryAugmentation, MemoryRecord, Request


class BaseMemory(ABC):
    @abstractmethod
    def prepare(self, records: Sequence[MemoryRecord], workspace: Path) -> None:
        """Ingest, index, or train once in the isolated episode workspace."""

    @abstractmethod
    def augment(self, request: Request) -> MemoryAugmentation:
        """Return context and/or an adapter without learning from predictions."""

    @abstractmethod
    def close(self) -> None:
        """Release resources, including after partially completed preparation."""
