import json

import pytest

from memory_bench.agents import AGENTS, BaseAgent, register_agent
from memory_bench.benchmarks import BENCHMARKS, BaseBenchmark, register_benchmark
from memory_bench.config import ConfigError, PluginError, PluginSpec, instantiate, resolve_class
from memory_bench.generators import GENERATORS, BaseGenerator, register_generator
from memory_bench.memories import MEMORIES, BaseMemory, register_memory
from memory_bench.registry import make_register
from memory_bench.runner import run


@pytest.mark.parametrize("register,registry,base,builtin", [
    (register_agent, AGENTS, BaseAgent, "shared"),
    (register_benchmark, BENCHMARKS, BaseBenchmark, "synthetic"),
    (register_generator, GENERATORS, BaseGenerator, "deterministic"),
    (register_memory, MEMORIES, BaseMemory, "none"),
])
def test_public_decorators_register_class_without_constructing_it(register, registry, base, builtin):
    class Additional(registry[builtin]):
        def __init__(self):
            pytest.fail("Registration and class lookup must not initialize backends")

    key = "test_decorated"
    assert key not in registry
    try:
        assert register(key)(Additional) is Additional
        assert resolve_class(PluginSpec("variant", key), registry, base) is Additional
    finally:
        registry.pop(key, None)


def test_decorator_rejects_duplicates_without_replacing_class():
    registry = {}
    register = make_register(registry, BaseMemory)
    original = MEMORIES["none"]
    register("memory")(original)
    with pytest.raises(ValueError, match="already registered"):
        register("memory")(MEMORIES["verbatim"])
    assert registry == {"memory": original}


@pytest.mark.parametrize("name", ["", "has spaces", "module:Class", None])
def test_decorator_rejects_names_that_cannot_be_selected_in_config(name):
    registry = {}
    with pytest.raises(ValueError, match="Registration name"):
        make_register(registry, BaseMemory)(name)
    assert registry == {}


def test_decorator_rejects_wrong_base_and_abstract_class():
    registry = {}
    register = make_register(registry, BaseMemory)
    with pytest.raises(TypeError, match="subclass of BaseMemory"):
        register("wrong")(BENCHMARKS["synthetic"])
    with pytest.raises(TypeError, match="abstract; implement"):
        register("incomplete")(BaseMemory)
    assert registry == {}


@pytest.mark.parametrize("base", [BaseAgent, BaseBenchmark, BaseGenerator, BaseMemory])
def test_base_classes_require_implementations(base):
    with pytest.raises(TypeError, match="abstract"):
        base()


def test_registries_reject_unrelated_and_incomplete_classes(monkeypatch):
    class Unrelated:
        pass

    class Incomplete(BaseMemory):
        def prepare(self, records, workspace):
            pass

    monkeypatch.setitem(MEMORIES, "wrong", Unrelated)
    monkeypatch.setitem(MEMORIES, "incomplete", Incomplete)
    with pytest.raises(PluginError, match="subclass of BaseMemory"):
        resolve_class(PluginSpec("wrong", "wrong"), MEMORIES, BaseMemory)
    with pytest.raises(PluginError, match="abstract; implement: augment, close"):
        instantiate(PluginSpec("incomplete", "incomplete"), MEMORIES, BaseMemory)


def test_names_default_to_types_and_repeated_names_are_rejected(config_factory):
    config = config_factory(benchmarks=[(None, "synthetic", {})], memories=[(None, "none", {})])
    assert config.benchmarks[0].name == "synthetic"
    assert config.memories[0].name == "none"
    assert config.agent.name == "shared"
    assert config.generators[config.agent.generation.generator].type == "deterministic"
    with pytest.raises(ConfigError, match="memories names must be unique"):
        config_factory(memories=[(None, "verbatim", {}), (None, "verbatim", {"separator": " "})])
    with pytest.raises(ConfigError, match="benchmarks names must be unique"):
        config_factory(benchmarks=[("duplicate", "synthetic", {}), ("duplicate", "synthetic", {})])


def test_memory_variants_pass_options_and_keep_distinct_results(config_factory):
    config = config_factory(memories=[
        ("lines", "verbatim", {"separator": "\n"}),
        ("joined", "verbatim", {"separator": " "}),
    ])
    output = run(config)
    summary = json.loads((output / "summary.json").read_text())
    assert {pair["memory"]: pair["metrics"]["exact_match"] for pair in summary["pairs"]} == {
        "lines": 1.0, "joined": 0.0,
    }
    assert summary["configuration"]["memories"] == [
        {"name": "lines", "type": "verbatim", "options": {"separator": "\n"}},
        {"name": "joined", "type": "verbatim", "options": {"separator": " "}},
    ]
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert all(row["case_id"] == row["request_id"] for row in rows)


@pytest.mark.parametrize("registry,base,component_type,options", [
    (MEMORIES, BaseMemory, "verbatim", {"not_an_option": 1}),
    (MEMORIES, BaseMemory, "verbatim", {"separator": 42}),
    (BENCHMARKS, BaseBenchmark, "synthetic", {"prefix": False}),
    (GENERATORS, BaseGenerator, "deterministic", {"unknown_answer": 42}),
    (AGENTS, BaseAgent, "shared", {"system_prompt": 42}),
])
def test_invalid_options_identify_component_and_option(registry, base, component_type, options):
    with pytest.raises(PluginError) as caught:
        instantiate(PluginSpec("experiment_variant", component_type, options), registry, base)
    assert "experiment_variant" in str(caught.value)
    assert component_type in str(caught.value)
    assert next(iter(options)) in str(caught.value)


def test_loader_only_benchmark_rejected_before_any_matrix_work(config_factory, monkeypatch, tmp_path):
    # Keep exercising the generic guard now that LME itself supports scoring.
    monkeypatch.setattr(BENCHMARKS["longmemeval_v2"], "supports_scoring", False)
    def unexpected(*args, **kwargs):
        pytest.fail("Scoring support must be checked before loading data or constructing backends")

    monkeypatch.setattr(BENCHMARKS["synthetic"], "load", unexpected)
    monkeypatch.setattr(BENCHMARKS["longmemeval_v2"], "load", unexpected)
    monkeypatch.setattr(MEMORIES["none"], "__init__", unexpected)
    monkeypatch.setattr(GENERATORS["deterministic"], "__init__", unexpected)
    config = config_factory(benchmarks=[
        ("demo", "synthetic", {}),
        ("lme", "longmemeval_v2", {"data_root": "not-needed-for-preflight"}),
    ])
    with pytest.raises(PluginError, match="does not support scoring.*inspection"):
        run(config)
    summary_path = next((tmp_path / "results").glob("*/summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "failed"
    assert summary["pairs"] == []
