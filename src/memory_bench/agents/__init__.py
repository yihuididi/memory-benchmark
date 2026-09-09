"""Agent registry. Import new implementation modules below to register them."""

from memory_bench.agents.base import BaseAgent
from memory_bench.registry import make_register

AGENTS: dict[str, type[BaseAgent]] = {}
register_agent = make_register(AGENTS, BaseAgent)

# Decorators must be defined before importing the implementations that use them.
from memory_bench.agents.shared import SharedAgent
from memory_bench.agents.longmemeval_v2 import LongMemEvalV2Agent
