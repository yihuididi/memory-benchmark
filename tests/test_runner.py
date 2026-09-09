from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_bench.config import ConfigError, PluginError, load_config
from memory_bench.benchmarks.synthetic import SyntheticBenchmark
from memory_bench.runner import run
from memory_bench.benchmarks.base import BaseBenchmark
from memory_bench.memories.base import BaseMemory
from memory_bench.generators.base import BaseGenerator
from memory_bench.types import (
    Episode,
    EvaluationCase,
    MemoryAugmentation,
    MemoryRecord,
    Prediction,
    Request,
)


def read_results(result_dir):
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (result_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return summary, predictions


def test_offline_matrix_runs_all_pairs_and_preserves_distinct_episode_answers(
    config_factory, tmp_path
):
    config = config_factory(
        benchmarks=[
            ("synthetic_a", "synthetic", {"prefix": "a"}),
            ("synthetic_b", "synthetic", {"prefix": "b"}),
        ],
        memories=[
            ("none", "none", {}),
            ("verbatim", "verbatim", {}),
        ],
    )

    result_dir = run(config)
    summary, predictions = read_results(result_dir)

    assert result_dir.resolve().is_relative_to(tmp_path / "results")
    assert summary["status"] == "completed"
    assert len(summary["pairs"]) == 4
    assert len(predictions) == 16
    assert len(
        {
            (row["benchmark"], row["memory"], row["episode_id"], row["request_id"])
            for row in predictions
        }
    ) == 16
    assert {(pair["benchmark"], pair["memory"]) for pair in summary["pairs"]} == {
        (benchmark, memory)
        for benchmark in ("synthetic_a", "synthetic_b")
        for memory in ("none", "verbatim")
    }
    for pair in summary["pairs"]:
        assert pair["metrics"]["exact_match"] == (1.0 if pair["memory"] == "verbatim" else 0.0)
        assert pair["episodes"] == 2
        assert pair["questions"] == 4
        assert pair["preparation_seconds"] >= 0
        assert pair["inference_seconds"] >= 0
    for row in predictions:
        assert row["scores"]["exact_match"] == (1.0 if row["memory"] == "verbatim" else 0.0)
        assert isinstance(row["answer"], str)
        assert row["preparation_seconds"] >= 0
        assert row["inference_seconds"] >= 0


def test_runner_isolates_episode_state_and_keeps_references_in_evaluator(
    config_factory, install_plugins, tmp_path
):
    memories, generations, references_scored = [], [], []

    class RecordingBenchmark(BaseBenchmark):
        def __init__(self, marker):
            assert marker == "benchmark-option"

        def load(self):
            for episode_index in range(2):
                yield Episode(
                    id=f"episode-{episode_index}",
                    records=(MemoryRecord("same-id", f"MEMORY-{episode_index}"),),
                    cases=tuple(
                        EvaluationCase(
                            request=Request(
                                f"request-{query_index}",
                                f"QUESTION-{episode_index}-{query_index}",
                            ),
                            reference=f"PRIVATE-REFERENCE-{episode_index}-{query_index}",
                        )
                        for query_index in range(2)
                    ),
                )

        def score(self, case, prediction):
            references_scored.append(case.reference)
            assert prediction.answer == "answer"
            return {"correct": 1.0}

        def aggregate(self, scores):
            scores = list(scores)
            return {"custom_total": sum(score["correct"] for score in scores)}

    class RecordingMemory(BaseMemory):
        def __init__(self, marker):
            assert marker == "memory-option"
            self.prepare_calls = 0
            self.requests = []
            self.closed = False
            memories.append(self)

        def prepare(self, records, workspace):
            self.prepare_calls += 1
            self.workspace = Path(workspace)
            assert self.workspace.is_dir()
            self.records = tuple(records)
            assert all(isinstance(record, MemoryRecord) for record in self.records)
            assert all("PRIVATE-REFERENCE" not in record.text for record in self.records)

        def augment(self, request):
            assert isinstance(request, Request)
            assert not hasattr(request, "reference")
            self.requests.append(request)
            return MemoryAugmentation(self.records[0].text)

        def close(self):
            self.closed = True

    class RecordingGeneration(BaseGenerator):
        def __init__(self, marker):
            assert marker == "generation-option"
            self.prompts = []
            self.closed = False
            generations.append(self)

        def generate(self, messages, *, settings, model_adapter=None, attachments=()):
            prompt = "\n".join(message.content for message in messages)
            assert "PRIVATE-REFERENCE" not in prompt
            assert settings["temperature"] == 0.0
            self.prompts.append(prompt)
            return Prediction("answer")

        def close(self):
            self.closed = True

    registered = install_plugins(
        Benchmark=RecordingBenchmark,
        Memory=RecordingMemory,
        Generation=RecordingGeneration,
    )
    config = config_factory(
        benchmarks=[("recording", registered["Benchmark"], {"marker": "benchmark-option"})],
        memories=[("recording", registered["Memory"], {"marker": "memory-option"})],
        generation=registered["Generation"],
        generation_options={"marker": "generation-option"},
    )

    summary, predictions = read_results(run(config))

    used_memories = [memory for memory in memories if memory.prepare_calls]
    used_generations = [generation for generation in generations if generation.prompts]
    assert len(used_memories) == len(used_generations) == 2
    assert len({memory.workspace for memory in used_memories}) == 2
    for index, memory in enumerate(used_memories):
        assert memory.prepare_calls == 1
        assert len(memory.requests) == 2
        assert memory.closed
        assert memory.workspace.resolve().is_relative_to(tmp_path / "artifacts")
        assert memory.records[0].text == f"MEMORY-{index}"
    for index, generation in enumerate(used_generations):
        assert len(generation.prompts) == 2
        assert generation.closed
        assert all(f"MEMORY-{index}" in prompt for prompt in generation.prompts)
        assert all(f"MEMORY-{1 - index}" not in prompt for prompt in generation.prompts)
        assert f"QUESTION-{index}-0" not in generation.prompts[1]
    assert len(references_scored) == len(set(references_scored)) == 4
    assert summary["pairs"][0]["metrics"] == {"custom_total": 4.0}
    assert len(predictions) == 4


def test_runtime_failure_preserves_predictions_and_original_error_during_cleanup(
    config_factory, install_plugins, tmp_path
):
    closed = []

    class FailingMemory(BaseMemory):
        def prepare(self, records, workspace):
            pass

        def augment(self, request):
            return MemoryAugmentation()

        def close(self):
            closed.append("memory")
            raise RuntimeError("memory cleanup exploded")

    class FailingGeneration(BaseGenerator):
        def __init__(self):
            self.calls = 0

        def generate(self, messages, *, settings, model_adapter=None, attachments=()):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("inference exploded")
            return Prediction("first completed answer")

        def close(self):
            closed.append("generation")
            raise RuntimeError("generation cleanup exploded")

    registered = install_plugins(Memory=FailingMemory, Generation=FailingGeneration)
    config = config_factory(
        memories=[("failing", registered["Memory"], {})],
        generation=registered["Generation"],
    )

    with pytest.raises(RuntimeError) as caught:
        run(config)

    assert str(caught.value) == "inference exploded"
    assert set(closed) == {"memory", "generation"}
    notes = "\n".join(getattr(caught.value, "__notes__", ()))
    assert "memory cleanup exploded" in notes
    assert "generation cleanup exploded" in notes
    summaries = list((tmp_path / "results").rglob("summary.json"))
    assert len(summaries) == 1
    summary, predictions = read_results(summaries[0].parent)
    assert summary["status"] == "failed"
    assert "inference exploded" in json.dumps(summary)
    assert len(predictions) == 1
    assert predictions[0]["answer"] == "first completed answer"


def test_generation_constructor_failure_closes_already_created_memory(
    config_factory, install_plugins, tmp_path
):
    closed = []

    class Memory(BaseMemory):
        def prepare(self, records, workspace):
            pytest.fail("Preparation must wait until generation construction succeeds")

        def augment(self, request):
            return MemoryAugmentation()

        def close(self):
            closed.append("memory")

    class Generation(BaseGenerator):
        def __init__(self):
            raise ConnectionError("cannot connect backend")

        def generate(self, messages, *, settings, model_adapter=None, attachments=()):
            pytest.fail("Generation construction should have failed")

        def close(self):
            pass

    registered = install_plugins(Memory=Memory, Generation=Generation)
    config = config_factory(
        memories=[("recording", registered["Memory"], {})],
        generation=registered["Generation"],
    )

    with pytest.raises(PluginError, match="cannot connect backend"):
        run(config)

    assert closed == ["memory"]
    summaries = list((tmp_path / "results").rglob("summary.json"))
    assert len(summaries) == 1
    summary, predictions = read_results(summaries[0].parent)
    assert summary["status"] == "failed"
    assert predictions == []


def test_invalid_aggregate_preserves_scored_predictions_and_writes_failed_summary(
    config_factory, install_plugins, tmp_path
):
    class InvalidAggregateBenchmark(SyntheticBenchmark):
        def aggregate(self, scores):
            return {"exact_match": float("nan")}

    registered = install_plugins(Benchmark=InvalidAggregateBenchmark)
    config = config_factory(
        benchmarks=[("invalid_aggregate", registered["Benchmark"], {})],
        memories=[("verbatim", "verbatim", {})],
    )

    with pytest.raises(ValueError, match="(?i)finite"):
        run(config)

    summaries = list((tmp_path / "results").rglob("summary.json"))
    assert len(summaries) == 1
    summary, predictions = read_results(summaries[0].parent)
    assert summary["status"] == "failed"
    assert len(predictions) == 4
    assert all(row["scores"] == {"exact_match": 1.0} for row in predictions)


def test_unknown_type_lists_available_registrations(config_factory):
    config = config_factory(memories=[("missing", "nonexistent", {})])
    with pytest.raises(PluginError, match="Unknown BaseMemory type 'nonexistent'") as caught:
        run(config)
    assert "Available types: none, verbatim" in str(caught.value)


def test_config_rejects_retired_class_key(tmp_path):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        '[generation]\nclass = "memory_bench.demos:DeterministicGenerator"\n'
        '[[benchmarks]]\ntype = "synthetic"\n'
        '[[memories]]\ntype = "none"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Unknown key.*class"):
        load_config(config_path)


def test_judge_failure_preserves_unscored_generation_and_closes_evaluator(config_factory, install_plugins):
    closed = []
    class FailingJudgeBenchmark(SyntheticBenchmark):
        def score(self, case, prediction):
            raise ValueError("Judge response could not be parsed")
        def close(self):
            closed.append(True)
    registered = install_plugins(Benchmark=FailingJudgeBenchmark)
    config = config_factory(benchmarks=[("judge_failure", registered["Benchmark"], {})])
    with pytest.raises(ValueError, match="Judge response"):
        run(config)
    output = next(config.results_dir.iterdir())
    rows = [json.loads(line) for line in (output / "generations.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["answer"] == "UNKNOWN"
    assert "scores" not in rows[0]
    assert (output / "predictions.jsonl").read_text() == ""
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"
    assert closed == [True, True]  # Configuration preflight and the actual run.
