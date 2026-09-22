# Memory Bench

A small Python harness for comparing memory backends across benchmarks. The
runtime uses only the Python standard library; development dependencies are
managed with [uv](https://docs.astral.sh/uv/).

## Quick start

Requirements: Python 3.12+ and uv.

```sh
uv sync --locked
uv run memory-bench run --config configs/longmemeval-v2-enterprise.toml
uv run pytest
```

The default enterprise configuration runs the LongMemEval v2 benchmark stack,
including the selected memory backends and the required model or judge setup.

Results are written to a unique directory under `results/` and episode-local
workspaces are written under `artifacts/`. Each run contains:

- `summary.json`: configuration, status, metrics, counts, and timings.
- `generations.jsonl`: answers persisted before evaluation, including an answer whose judge failed.
- `predictions.jsonl`: one scored prediction per completed question, with evaluator diagnostics.

Run the CLI offline after dependencies are installed:

```sh
uv run --offline --no-sync memory-bench run --config configs/longmemeval-v2-enterprise.toml
```

## Configuration

### Docker Compose

Select the experiment in `docker-compose.yaml` under `command`. Set
`build.args.BENCH_EXTRA` to the space-separated optional dependency extras
needed by that experiment, then build and run:

```yaml
build:
  context: .
  args:
    BENCH_EXTRA: "openai-compatible local-transformers"
```

```sh
docker compose up --build
```

The current enterprise configuration uses `openai-compatible` for the model server
client and `local-transformers` for its local evaluator. Add `transformers-vision` when using the
LongMemEval agent with memory context that requires its vision processor:
`BENCH_EXTRA: "openai-compatible local-transformers transformers-vision"`. These names refer to
`[project.optional-dependencies]` in `pyproject.toml`; each becomes a separate
`uv sync --extra` argument.
Rebuild the image whenever you change the extras.

### Experiment configuration

Start with [`configs/longmemeval-v2-enterprise.toml`](configs/longmemeval-v2-enterprise.toml). A configuration contains one or more benchmark and memory entries, plus an agent and named generators:

```toml
[[benchmarks]]
type = "synthetic"

[[memories]]
name = "verbatim"
type = "verbatim"

[agent]
type = "shared"

[generators.reader]
type = "deterministic"

[agent.generation]
generator = "reader"

[agent.generation.settings]
temperature = 0.0
```

`type` selects a registered implementation. `name` is an optional unique label
used in results and defaults to `type`. Pass constructor arguments in an
`options` table. Paths are relative to the process working directory.

Each `[generators.<name>]` defines a backend `type` and constructor `options`.
Select it with `generator = "<name>"` under `[agent.generation]`,
`[benchmarks.generation]`, or `[memories.generation]`. Each binding has its own
`settings` table; settings are never inherited from another component. Agent
bindings are required. Benchmarks and memories can omit generation when unused.
LongMemEval requires a benchmark binding when selected questions need LLM judging.

To use the reader endpoint for judging, set `generator = "reader"` in
`[benchmarks.generation]`, with separate judge settings such as `max_tokens = 512`
and `temperature = 0`. See `configs/longmemeval-v2-enterprise.toml` for a separate
local Transformers judge. Its output is constrained to the benchmark JSON schema;
remote output is validated against that same schema. Invalid judgments fail the run.

The runner creates a benchmark and its generator handle per benchmark-memory pair,
and fresh memory, agent, and memory/agent generator handles per episode. Backends
load lazily. Reusing a definition does not share backend instances, mutable state,
or adapters; two local bindings may load two copies of model weights. Memory is
prepared once per episode. Only benchmark scoring receives evaluation references.
The runner closes generators, including on failure; components borrow them.

Migration: the old `[generation]` and `[benchmarks.options.evaluator]` tables are
rejected. Move model construction options into named definitions, and move
inference settings into component bindings. Replace evaluator `backend` with
generator `type`. For Transformers, `max_new_tokens`, `do_sample`, and
`chat_template_kwargs` belong in binding settings; `model_path`, `device`, `dtype`,
and optional `max_context_tokens` belong in backend options. Set
`chat_template_kwargs.enable_thinking = false` to preserve the former Qwen judge
behavior. Remote generators require an explicit `base_url` and `model`.

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

- Benchmarks: `longmemeval_v2`, `ripple_edit`.
- Memories: `none`, `verbatim`.
- Generators: `deterministic`, `openai_compatible` (user-supplied model endpoint).
- Agents: `longmemeval_v2` (upstream prompts and evidence budget), `ripple_edit`.

There is no automatic module scanning. Every new implementation needs an
explicit import in its package `__init__.py`, and can then be selected in TOML
with its registered `type` name.

### Third-party code

This project includes or adapts code from third-party sources.
Copyright and licensing information for those components is available in
[NOTICE.md](NOTICE.md) and/or the [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/)
directory.

Benchmarks and memories receive optional dependencies through
`bind_generation(generator, settings)` before validation/preparation. Implementations
can use `self.generation.generate(..., settings=self.generation_settings)` during
preparation, retrieval, or scoring. Do not close borrowed generators. Agents retain
`answer(request, memory, generation, settings)`.

Generators may implement `preflight()` for checks that do not load weights or
contact servers. `generate_structured(messages, settings=..., schema=...)` defaults
to ordinary generation; callers must validate the returned JSON. The Transformers
backend overrides it with constrained decoding. Its ordinary generation supports
text only and explicitly rejects attachments and model adapters.
