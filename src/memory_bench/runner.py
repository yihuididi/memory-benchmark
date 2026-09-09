"""Sequential experiment matrix with episode isolation and durable partial results."""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from memory_bench.agents import AGENTS, BaseAgent
from memory_bench.benchmarks import BENCHMARKS, BaseBenchmark
from memory_bench.config import PluginError, RunConfig, instantiate, resolve_class
from memory_bench.generators import GENERATORS, BaseGenerator
from memory_bench.memories import MEMORIES, BaseMemory


@contextmanager
def _resources() -> Iterator[list[Any]]:
    """Close every successfully constructed resource, preserving the original error."""
    resources: list[Any] = []
    original: BaseException | None = None
    try:
        yield resources
    except BaseException as exc:
        original = exc
        raise
    finally:
        cleanup_errors: list[Exception] = []
        for resource in reversed(resources):
            try:
                resource.close()
            except Exception as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if original is not None:
                for exc in cleanup_errors:
                    original.add_note(f"Resource cleanup also failed: {exc}")
            else:
                first, *rest = cleanup_errors
                for exc in rest:
                    first.add_note(f"Resource cleanup also failed: {exc}")
                raise first


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _metrics(value: Any) -> dict[str, float]:
    metrics = dict(value)
    for key, score in metrics.items():
        if (
            not isinstance(key, str)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError("Benchmark metrics must map string names to finite numbers")
    return metrics


def run(config: RunConfig) -> Path:
    """Run all benchmark/memory combinations; return the unique results directory."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    output = config.results_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    artifacts = config.artifacts_dir / run_id
    summary: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "configuration": copy.deepcopy(config.snapshot),
        "artifacts_dir": str(artifacts),
        "pairs": [],
    }
    started = time.perf_counter()
    summary_path = output / "summary.json"
    _write_summary(summary_path, summary)
    try:
        # Check the entire matrix before any dataset loading, training, or inference.
        for spec in config.benchmarks:
            benchmark_class = resolve_class(spec, BENCHMARKS, BaseBenchmark)
            if not benchmark_class.supports_scoring:
                raise PluginError(
                    f"Benchmark '{spec.name}' (type '{spec.type}') does not support scoring yet. "
                    "It is available for data inspection only."
                )
        for spec in config.memories:
            resolve_class(spec, MEMORIES, BaseMemory)
        resolve_class(config.agent, AGENTS, BaseAgent)
        resolve_class(config.generation, GENERATORS, BaseGenerator)
        for spec in config.benchmarks:
            with _resources() as resources:
                benchmark = instantiate(spec, BENCHMARKS, BaseBenchmark)
                resources.append(benchmark)
                benchmark.validate_run(config.agent, config.generation)
        artifacts.mkdir(parents=True, exist_ok=False)
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions, \
                (output / "generations.jsonl").open("w", encoding="utf-8") as generations:
            for benchmark_spec in config.benchmarks:
                for memory_spec in config.memories:
                    with _resources() as pair_resources:
                        benchmark = instantiate(benchmark_spec, BENCHMARKS, BaseBenchmark)
                        pair_resources.append(benchmark)
                        benchmark.validate_run(config.agent, config.generation)
                        pair: dict[str, Any] = {
                            "benchmark": benchmark_spec.name,
                            "memory": memory_spec.name,
                            "status": "running",
                            "metrics": {},
                            "episodes": 0,
                            "questions": 0,
                            "preparation_seconds": 0.0,
                            "inference_seconds": 0.0,
                        }
                        summary["pairs"].append(pair)
                        scores: list[dict[str, float]] = []
                        episode_ids: set[str] = set()
                        for episode_index, episode in enumerate(benchmark.load()):
                            if episode.id in episode_ids:
                                raise ValueError(f"Duplicate episode id {episode.id!r} in {benchmark_spec.name}")
                            episode_ids.add(episode.id)
                            # Use numeric directories: dataset IDs may contain slashes or other path characters.
                            workspace = artifacts / benchmark_spec.name / memory_spec.name / str(episode_index)
                            workspace.mkdir(parents=True)
                            with _resources() as resources:
                                memory = instantiate(memory_spec, MEMORIES, BaseMemory)
                                resources.append(memory)
                                generation = instantiate(config.generation, GENERATORS, BaseGenerator)
                                resources.append(generation)
                                agent = instantiate(config.agent, AGENTS, BaseAgent)
                                before = time.perf_counter()
                                # Never supply cases/references to memory preparation.
                                memory.prepare(copy.deepcopy(episode.records), workspace)
                                preparation_seconds = time.perf_counter() - before
                                pair["preparation_seconds"] += preparation_seconds
                                request_ids: set[str] = set()
                                for case in episode.cases:
                                    if case.request.id in request_ids:
                                        raise ValueError(f"Duplicate request id {case.request.id!r} in episode {episode.id!r}")
                                    request_ids.add(case.request.id)
                                    before = time.perf_counter()
                                    prediction = agent.answer(
                                        copy.deepcopy(case.request), memory, generation,
                                        copy.deepcopy(config.generation_settings),
                                    )
                                    inference_seconds = time.perf_counter() - before
                                    row = {
                                        "benchmark": benchmark_spec.name,
                                        "memory": memory_spec.name,
                                        "episode_id": episode.id,
                                        "request_id": case.request.id,
                                        "case_id": case.id if case.id is not None else case.request.id,
                                        "answer": prediction.answer,
                                        "metadata": prediction.metadata,
                                        "preparation_seconds": preparation_seconds,
                                        "inference_seconds": inference_seconds,
                                    }
                                    # Persist the generated answer before a judge can fail.
                                    generations.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                                    generations.flush()
                                    case_scores = _metrics(benchmark.score(case, prediction))
                                    row["scores"] = case_scores
                                    row["evaluation"] = benchmark.evaluation_details()
                                    predictions.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                                    predictions.flush()
                                    scores.append(case_scores)
                                    pair["questions"] += 1
                                    pair["inference_seconds"] += inference_seconds
                                pair["episodes"] += 1
                            _write_summary(summary_path, summary)
                        if not scores:
                            raise ValueError(f"Benchmark '{benchmark_spec.name}' produced no evaluation questions")
                        pair["metrics"] = _metrics(benchmark.aggregate(scores))
                        pair["status"] = "completed"
                        _write_summary(summary_path, summary)
        summary["status"] = "completed"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if notes := getattr(exc, "__notes__", None):
            summary["error"]["notes"] = notes
        for pair in summary["pairs"]:
            if pair["status"] == "running":
                pair["status"] = "failed"
        exc.add_note(f"Partial results: {output}")
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        original = sys.exception()
        try:
            _write_summary(summary_path, summary)
        except Exception as exc:
            if original is None:
                raise
            original.add_note(f"Could not save final summary: {exc}")
    return output
