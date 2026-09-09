"""Memory registry. Import new implementation modules below to register them."""

from memory_bench.memories.base import BaseMemory
from memory_bench.registry import make_register

MEMORIES: dict[str, type[BaseMemory]] = {}
register_memory = make_register(MEMORIES, BaseMemory)

# Decorators must be defined before importing the implementations that use them.
from memory_bench.memories.no_memory import NoMemory
from memory_bench.memories.verbatim import VerbatimMemory
