from __future__ import annotations

import json
import time

import pytest

from memory_bench.config import load_config
from memory_bench.benchmarks import BENCHMARKS
from memory_bench.memories import MEMORIES
from memory_bench.generators import GENERATORS
from memory_bench.agents import AGENTS
from memory_bench.agents import register_agent
from memory_bench.agents.longmemeval_v2 import LongMemEvalV2Agent
from memory_bench.types import Message


class _FixtureTokenCount:
    shape = (1, 1)


class _FixtureProcessor:
    def apply_chat_template(self, *args, **kwargs):
        return "fixture"

    def __call__(self, **kwargs):
        return {"input_ids": _FixtureTokenCount()}


@register_agent("fixture_agent")
class FixtureAgent(LongMemEvalV2Agent):
    def _get_processor(self):
        return _FixtureProcessor()

    def answer(self, request, memory, generation, settings):
        started = time.perf_counter()
        augmentation = memory.augment(request)
        memory_seconds = time.perf_counter() - started
        context = "\n".join(item["value"] for item in augmentation.context_items)
        if augmentation.context:
            context = f"{augmentation.context}\n{context}" if context else augmentation.context
        prediction = generation.generate(
            (
                Message("system", "Answer using the supplied memory context."),
                Message("user", f"Memory context:\n{context}"),
                Message("user", f"Question:\n{request.question}"),
            ),
            settings=settings,
            model_adapter=augmentation.model_adapter,
            attachments=request.attachments,
        )
        metadata = {
            **prediction.metadata,
            "timings": {
                "memory_query_duration_seconds": memory_seconds,
                "reader_duration_seconds": time.perf_counter() - started,
            },
        }
        return type(prediction)(prediction.answer, metadata)


@pytest.fixture
def install_plugins(monkeypatch):
    """Temporarily register test implementations through the public registries."""
    registries = {"Benchmark": BENCHMARKS, "Memory": MEMORIES, "Generation": GENERATORS, "Agent": AGENTS}

    def install(**classes):
        registered = {}
        for name, cls in classes.items():
            key = f"test_{name.lower()}"
            monkeypatch.setitem(registries[name], key, cls)
            registered[name] = key
        return registered

    return install


@pytest.fixture
def config_factory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    def make(
        *,
        benchmarks=None,
        memories=None,
        generation="openai_compatible",
        generation_options=None,
        agent_type="longmemeval_v2",
        agent_options=None,
    ):
        benchmarks = benchmarks or [("longmemeval", "longmemeval_v2", {
            "data_root": "data/longmemeval-v2", "domain": "enterprise", "tier": "small",
        })]
        memories = memories or [("none", "none", {})]

        def options_line(options):
            entries = ", ".join(
                f"{key} = {json.dumps(value)}" for key, value in options.items()
            )
            return f"options = {{{entries}}}"

        lines = [
            "[run]",
            'results_dir = "results"',
            'artifacts_dir = "artifacts"',
            "[agent]",
            f"type = {json.dumps(agent_type)}",
            '[agent.options]',
            *(
                f"{key} = {json.dumps(value)}"
                for key, value in (agent_options or {"domain": "enterprise"}).items()
            ),
            "[generators.reader]",
            f"type = {json.dumps(generation)}",
            options_line(
                generation_options
                if generation_options is not None
                else ({
                    "model": "test-model", "base_url": "http://127.0.0.1:8001/v1",
                } if generation == "openai_compatible" else {})
            ),
            "[agent.generation]",
            'generator = "reader"',
            "[agent.generation.settings]",
            "temperature = 0.0",
        ]
        for section, entries in (("benchmarks", benchmarks), ("memories", memories)):
            for name, component_type, options in entries:
                options = dict(options)
                lines.extend(
                    [
                        f"[[{section}]]",
                        *([f"name = {json.dumps(name)}"] if name is not None else []),
                        f"type = {json.dumps(component_type)}",
                        options_line(options),
                    ]
                )
        path = config_dir / "run.toml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return load_config(path)

    return make
