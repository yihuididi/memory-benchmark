from collections.abc import Sequence
from pathlib import Path
from memory_bench.memories import register_memory
from memory_bench.memories.base import BaseMemory
from memory_bench.types import MemoryAugmentation, MemoryRecord, Request


@register_memory("none")
class NoMemory(BaseMemory):
    def prepare(self, records: Sequence[MemoryRecord], workspace: Path) -> None:
        pass

    def augment(self, request: Request) -> MemoryAugmentation:
        return MemoryAugmentation()

    def close(self) -> None:
        pass
