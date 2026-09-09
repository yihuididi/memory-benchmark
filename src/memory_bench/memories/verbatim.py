from collections.abc import Sequence
from pathlib import Path
from memory_bench.memories import register_memory
from memory_bench.memories.base import BaseMemory
from memory_bench.types import MemoryAugmentation, MemoryRecord, Request


@register_memory("verbatim")
class VerbatimMemory(BaseMemory):
    """Return all episode records as context; deliberately no retrieval logic."""

    def __init__(self, *, separator: str = "\n") -> None:
        if not isinstance(separator, str):
            raise ValueError("separator must be a string")
        self.separator = separator
        self._context = ""

    def prepare(self, records: Sequence[MemoryRecord], workspace: Path) -> None:
        self._context = self.separator.join(record.text for record in records)

    def augment(self, request: Request) -> MemoryAugmentation:
        return MemoryAugmentation(context=self._context)

    def close(self) -> None:
        self._context = ""
