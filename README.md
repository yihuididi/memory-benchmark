# Memory Bench

A small Python harness for comparing memory backends across benchmarks. The
runtime uses only the Python standard library; development dependencies are
managed with [uv](https://docs.astral.sh/uv/).

## Quick start

Requirements: Python 3.12+ and uv.

```sh
uv sync --locked
uv run memory-bench run --config configs/demo.toml
uv run pytest
```

The demo runs every configured benchmark and memory combination. It uses a
synthetic benchmark, a no-memory baseline, a verbatim memory, and a
deterministic generator, so it needs no dataset, model, credentials, or GPU.

Results are written to a unique directory under `results/` and episode-local
workspaces are written under `artifacts/`. Each run contains:

- `summary.json`: configuration, status, metrics, counts, and timings.
- `generations.jsonl`: answers persisted before evaluation, including an answer whose judge failed.
- `predictions.jsonl`: one scored prediction per completed question, with evaluator diagnostics.

Run the CLI offline after dependencies are installed:

```sh
uv run --offline --no-sync memory-bench run --config configs/demo.toml
```

## Configuration

Start with [`configs/demo.toml`](configs/demo.toml). A configuration contains
one or more benchmark and memory entries, plus an agent and generator:

```toml
[[benchmarks]]
type = "synthetic"

[[memories]]
name = "verbatim"
type = "verbatim"

[agent]
type = "shared"

[generation]
type = "deterministic"

[generation.settings]
temperature = 0.0
```

`type` selects a registered implementation. `name` is an optional unique label
used in results and defaults to `type`. Pass constructor arguments in an
`options` table. Paths are relative to the process working directory.

The runner creates a fresh benchmark, memory, generator, and agent for each
benchmark-memory pair and episode. Memory is prepared once per episode, then
each question is evaluated independently. Evaluation references are available
only to the benchmark scorer, not to memory or the agent.

## Add a module

Benchmarks, memories, generators, and agents are added using the same explicit
registry pattern:

1. Create a module in the matching package under `src/memory_bench/`.
2. Subclass the package's base class and implement its required methods.
3. Register the implementation with a unique `type` name.
4. Import the module in that package's `__init__.py` after the registry and
    decorator definitions.

Use the supplied `workspace` for episode-local state, treat records as
read-only, and do not use predictions to update memory. Optional dependencies
should be imported inside the implementation so the demo remains usable.

### Memory example

Create a module under `src/memory_bench/memories/`, subclass `BaseMemory`, and
register it. Then import it in `memories/__init__.py`.

```python
from pathlib import Path
from collections.abc import Sequence

from memory_bench.memories import BaseMemory, register_memory
from memory_bench.types import MemoryAugmentation, MemoryRecord, Request


@register_memory("my_memory")
class MyMemory(BaseMemory):
    def __init__(self, separator: str = "\n"):
        self.separator = separator
        self.context = ""

    def prepare(self, records: Sequence[MemoryRecord], workspace: Path) -> None:
        self.context = self.separator.join(record.text for record in records)

    def augment(self, request: Request) -> MemoryAugmentation:
        return MemoryAugmentation(context=self.context)

    def close(self) -> None:
        self.context = ""
```

Add its import after the registry and decorator definitions:

```python
from memory_bench.memories.my_memory import MyMemory
```

Select it in TOML with `type = "my_memory"`.

### Benchmark example

Create a module under `src/memory_bench/benchmarks/`, subclass `BaseBenchmark`,
and register it. Import the module in `benchmarks/__init__.py`.

```python
from collections.abc import Iterable, Mapping, Sequence

from memory_bench.benchmarks import BaseBenchmark, register_benchmark
from memory_bench.types import Episode, EvaluationCase, Prediction


@register_benchmark("my_benchmark")
class MyBenchmark(BaseBenchmark):
    def load(self) -> Iterable[Episode]:
        ...

    def score(self, case: EvaluationCase, prediction: Prediction) -> dict[str, float]:
        ...

    def aggregate(self, scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
        ...
```

`load()` yields `Episode` objects containing `MemoryRecord` values and
`EvaluationCase` values. `score()` returns finite numeric metrics, and
`aggregate()` combines those question-level scores. Keep public request data in
`Request`; references and evaluator-only metadata belong in `EvaluationCase`.

For a generator, implement `BaseGenerator.generate()` and `close()` and use
`register_generator`. For an agent, implement `BaseAgent.answer()` and use
`register_agent`. Their modules belong under `src/memory_bench/generators/` and
`src/memory_bench/agents/`, respectively.

The built-in registry types are:

- Benchmarks: `synthetic`, `longmemeval_v2`.
- Memories: `none`, `verbatim`.
- Generators: `deterministic`, `openai_compatible` (user-supplied model endpoint).
- Agents: `shared`, `longmemeval_v2` (optional upstream prompts and evidence budget).

There is no automatic module scanning. Every new implementation needs an
explicit import in its package `__init__.py`, and can then be selected in TOML
with its registered `type` name.

### Third-party code

This project includes or adapts code from third-party sources.
Copyright and licensing information for those components is available in
[NOTICE.md](NOTICE.md) and/or the [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/)
directory.
