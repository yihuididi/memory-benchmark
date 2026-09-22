"""Generator registry. Import new implementation modules below to register them."""

from memory_bench.generators.base import BaseGenerator
from memory_bench.registry import make_register

GENERATORS: dict[str, type[BaseGenerator]] = {}
register_generator = make_register(GENERATORS, BaseGenerator)

# Decorators must be defined before importing the implementations that use them.
from memory_bench.generators.openai_compatible import OpenAICompatibleGenerator
from memory_bench.generators.transformers import TransformersGenerator
