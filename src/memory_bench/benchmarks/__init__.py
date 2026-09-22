"""Benchmark registry. Import new implementation modules below to register them."""

from memory_bench.benchmarks.base import BaseBenchmark
from memory_bench.registry import make_register

BENCHMARKS: dict[str, type[BaseBenchmark]] = {}
register_benchmark = make_register(BENCHMARKS, BaseBenchmark)

# Decorators must be defined before importing the implementations that use them.
from memory_bench.benchmarks.longmemeval_v2 import LongMemEvalV2Benchmark
