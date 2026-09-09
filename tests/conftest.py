from __future__ import annotations

import json

import pytest

from memory_bench.config import load_config
from memory_bench.benchmarks import BENCHMARKS
from memory_bench.memories import MEMORIES
from memory_bench.generators import GENERATORS
from memory_bench.agents import AGENTS


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
        generation="deterministic",
        generation_options=None,
    ):
        benchmarks = benchmarks or [
            ("synthetic", "synthetic", {})
        ]
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
            'type = "shared"',
            "[generation]",
            f"type = {json.dumps(generation)}",
            options_line(generation_options or {}),
            "[generation.settings]",
            "temperature = 0.0",
        ]
        for section, entries in (("benchmarks", benchmarks), ("memories", memories)):
            for name, component_type, options in entries:
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
