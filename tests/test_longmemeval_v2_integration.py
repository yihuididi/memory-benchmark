import json
from dataclasses import replace

import pytest

from test_longmemeval_v2 import dataset, read_jsonl, write_jsonl
from memory_bench.benchmarks.longmemeval_v2 import LongMemEvalV2Benchmark
from memory_bench.config import PluginSpec, ComponentSpec, GenerationBinding
from memory_bench.generators.base import BaseGenerator
from memory_bench.runner import run
from memory_bench.types import Prediction


def prepare_scoring_questions(dataset):
    path = dataset / "questions.jsonl"
    rows = read_jsonl(path)
    for row in rows:
        row.update(question_type="static-environment", eval_function="mc_choice_match", answer="A")
    rows[1].update(question_type="errors-gotchas", eval_function="llm_gotchas_checker", answer="PRIVATE INSIGHT")
    write_jsonl(path, rows)


def test_complete_matrix_preserves_judge_privacy_timing_and_diagnostics(
    dataset, config_factory, install_plugins, monkeypatch,
):
    prepare_scoring_questions(dataset)
    judge_calls, closes = [], []
    class RecordingJudge(BaseGenerator):
        def generate(self, messages, *, settings, **kwargs):
            judge_calls.append(messages)
            return Prediction('{"label": 1, "reason": "Matches reference"}')
        def close(self):
            closes.append(True)
    from memory_bench.generators import GENERATORS
    monkeypatch.setitem(GENERATORS, "fixture_judge", RecordingJudge)
    class Generator(BaseGenerator):
        def generate(self, messages, *, settings, model_adapter=None, attachments=()):
            text = json.dumps([m.content for m in messages])
            assert "PRIVATE" not in text
            assert "q-web" not in text
            assert "eval_function" not in text
            return Prediction(r"Explanation. \boxed{A}")
        def close(self):
            pass
    registered = install_plugins(Generation=Generator)
    config = config_factory(generation=registered["Generation"], memories=[("none", "none", {}),
                                                                                 ("verbatim", "verbatim", {})])
    config = replace(config, generators={**config.generators, "judge": PluginSpec("judge", "fixture_judge")},
        benchmarks=(ComponentSpec("lme", "longmemeval_v2", {
        "data_root": str(dataset),
    }, GenerationBinding("judge")),))
    output = run(config)
    summary = json.loads((output / "summary.json").read_text())
    predictions = read_jsonl(output / "predictions.jsonl")
    assert summary["status"] == "completed"
    assert len(predictions) == 8
    assert len(read_jsonl(output / "generations.jsonl")) == 8
    assert len(judge_calls) == 2
    assert all("PRIVATE INSIGHT" in json.dumps([m.content for m in call]) for call in judge_calls)
    assert len(closes) == 3  # Preflight, then one judge per memory variant.
    for pair in summary["pairs"]:
        assert pair["metrics"]["overall_full_set"] == 1
        assert pair["metrics"]["memory_query_avg_seconds"] >= 0
        assert pair["metrics"]["reader.avg_seconds"] >= 0
    assert all(row["evaluation"]["response_parsed_boxed"] == "A" for row in predictions)
    assert sum("judge" in row["evaluation"] for row in predictions) == 2


def test_missing_judge_fails_before_any_model_or_memory_work(dataset, config_factory, monkeypatch):
    prepare_scoring_questions(dataset)
    from memory_bench.generators import GENERATORS
    from memory_bench.memories import MEMORIES
    def unexpected(*a, **kw):
        pytest.fail("Invalid evaluator configuration must be detected before inference/preparation")
    monkeypatch.setattr(GENERATORS["openai_compatible"], "__init__", unexpected)
    monkeypatch.setattr(MEMORIES["none"], "__init__", unexpected)
    config = config_factory(generation="openai_compatible", benchmarks=[
        ("lme", "longmemeval_v2", {"data_root": str(dataset)}),
    ])
    with pytest.raises(ValueError, match="Configure benchmarks.generation"):
        run(config)


def test_deterministic_subset_needs_no_judge_or_model_dependencies(dataset):
    prepare_scoring_questions(dataset)
    benchmark = LongMemEvalV2Benchmark(data_root=dataset, limit=1)
    benchmark.validate_run(PluginSpec("agent", "shared"), PluginSpec("generator", "custom"))
    case = next(iter(benchmark.load())).cases[0]
    assert benchmark.score(case, Prediction(r"\boxed{A}"))["score"] == 1
    assert benchmark._judge is None


def test_dataset_cannot_override_judge_during_preflight(dataset):
    prepare_scoring_questions(dataset)
    rows = read_jsonl(dataset / "questions.jsonl")
    rows[0]["eval_function"] = "llm_gotchas_checker|evaluator_model=attacker"
    write_jsonl(dataset / "questions.jsonl", rows)
    benchmark = LongMemEvalV2Benchmark(data_root=dataset)
    with pytest.raises(ValueError, match="Unsupported option"):
        benchmark.validate_run(PluginSpec("agent", "shared"), PluginSpec("generator", "custom"))


def test_bound_judge_failure_preserves_answer_and_closes_backends(dataset, config_factory, install_plugins, monkeypatch):
    prepare_scoring_questions(dataset)
    closed, readers = [], []
    class Reader(BaseGenerator):
        def __init__(self):
            readers.append(self)
        def generate(self, *args, **kwargs):
            return Prediction(r"\boxed{A}")
        def close(self):
            closed.append("reader")
    class BrokenJudge(BaseGenerator):
        def generate(self, *args, **kwargs):
            return Prediction("incomplete judgment")
        def close(self):
            closed.append("judge")
    from memory_bench.generators import GENERATORS
    monkeypatch.setitem(GENERATORS, "broken_judge", BrokenJudge)
    registered = install_plugins(Generation=Reader)
    config = config_factory(generation=registered["Generation"])
    config = replace(config,
        generators={**config.generators, "judge": PluginSpec("judge", "broken_judge")},
        benchmarks=(ComponentSpec("lme", "longmemeval_v2", {"data_root": str(dataset)}, GenerationBinding("judge")),))
    with pytest.raises(ValueError, match="Could not parse"):
        run(config)
    output = next(config.results_dir.iterdir())
    generations = read_jsonl(output / "generations.jsonl")
    predictions = read_jsonl(output / "predictions.jsonl")
    assert len(generations) == len(predictions) + 1
    assert generations[-1]["answer"] == r"\boxed{A}"
    assert predictions
    assert closed.count("judge") == 2  # Preflight and pair-scoped judge.
    assert closed.count("reader") == len(readers)
