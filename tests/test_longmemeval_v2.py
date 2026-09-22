from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from uuid import UUID

import pytest

from memory_bench.benchmarks import BENCHMARKS
from memory_bench.cli import main
from memory_bench.config import PluginSpec, instantiate
from memory_bench.benchmarks.base import BaseBenchmark
from memory_bench.benchmarks.longmemeval_v2 import LongMemEvalV2Benchmark
from memory_bench.runner import run
from memory_bench.types import Prediction


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "longmemeval-v2"
    (root / "haystacks").mkdir(parents=True)
    (root / "question_screenshots").mkdir()
    # A relative symlink mirrors the upstream preparation script's media layout.
    (root / "question_screenshots" / "original.png").write_bytes(b"fake png")
    (root / "question_screenshots" / "question.png").symlink_to("original.png")

    questions = []
    for question_id, domain, image in (
        ("q-web-first", "web", "question_screenshots/question.png"),
        ("q-web-reversed", "web", None),
        ("q-enterprise", "enterprise", None),
        ("q-web-last", "web", None),
    ):
        questions.append(
            {
                "id": question_id,
                "domain": domain,
                "environment": "browser" if domain == "web" else "workspace",
                "question_type": "PRIVATE-QUESTION-CATEGORY",
                "question": f"What happened during {domain} browsing?",
                "image": image,
                "answer": f"PRIVATE-REFERENCE-{question_id}",
                "eval_function": "PRIVATE-EVALUATOR",
            }
        )
    write_jsonl(root / "questions.jsonl", questions)

    trajectories = []
    # Source order deliberately differs from haystack order.
    for trajectory_id, domain in (
        ("web-b", "web"),
        ("enterprise-a", "enterprise"),
        ("web-a", "web"),
        ("unused", "web"),
    ):
        media = root / "screenshots" / trajectory_id
        media.mkdir(parents=True)
        (media / "0.png").write_bytes(b"fake png")
        trajectories.append(
            {
                "id": trajectory_id,
                "domain": domain,
                "environment": "browser" if domain == "web" else "workspace",
                "goal": f"Find the catalog for {trajectory_id}",
                "outcome": "success",
                "start_url": "https://example.test/start",
                "states": [
                    {
                        "state_index": 0,
                        "step": 0,
                        "url": "https://example.test/start",
                        "action": None,
                        "thought": None,
                        "accessibility_tree": "INITIAL-PAGE catalog link",
                        "screenshot": f"screenshots/{trajectory_id}/0.png",
                    },
                    {
                        "state_index": 1,
                        "step": 7,
                        "url": "https://example.test/catalog",
                        "action": "click catalog",
                        "thought": "The catalog contains the required item",
                        "accessibility_tree": "FINAL-PAGE item found",
                        "screenshot": None,
                    },
                ],
            }
        )
    write_jsonl(root / "trajectories.jsonl", trajectories)

    haystacks = {
        "q-web-first": ["web-a", "web-b"],
        "q-web-reversed": ["web-b", "web-a"],
        "q-enterprise": ["enterprise-a"],
        "q-web-last": ["web-a", "web-b"],
    }
    (root / "haystacks" / "lme_v2_small.json").write_text(json.dumps(haystacks))
    medium = dict(haystacks, **{"q-web-first": ["web-b"], "q-web-last": ["web-b"]})
    (root / "haystacks" / "lme_v2_medium.json").write_text(json.dumps(medium))
    return root


def test_groups_identical_ordered_haystacks_and_preserves_question_order(dataset):
    benchmark = LongMemEvalV2Benchmark(data_root=dataset)
    episodes = list(benchmark.load())

    assert [[record.id for record in episode.records] for episode in episodes] == [
        ["web-a", "web-b"],
        ["web-b", "web-a"],
        ["enterprise-a"],
    ]
    assert [[case.metadata["id"] for case in episode.cases] for episode in episodes] == [
        ["q-web-first", "q-web-last"],
        ["q-web-reversed"],
        ["q-enterprise"],
    ]
    assert len({episode.id for episode in episodes}) == 3
    assert [episode.id for episode in benchmark.load()] == [episode.id for episode in episodes]
    # A filter must not change the identity of an otherwise identical haystack.
    limited = list(LongMemEvalV2Benchmark(data_root=dataset, limit=1).load())
    assert limited[0].id == episodes[0].id


def test_renders_trajectory_and_preserves_structured_states_with_resolved_images(dataset):
    record = list(LongMemEvalV2Benchmark(data_root=dataset, limit=1).load())[0].records[0]
    original = next(row for row in read_jsonl(dataset / "trajectories.jsonl") if row["id"] == "web-a")
    expected = deepcopy(original)
    expected["states"][0]["screenshot"] = str((dataset / "screenshots/web-a/0.png").resolve())

    assert record.metadata == expected
    for value in (
        original["goal"],
        original["outcome"],
        original["start_url"],
        "https://example.test/catalog",
        "click catalog",
        "The catalog contains the required item",
        "INITIAL-PAGE catalog link",
        "FINAL-PAGE item found",
        "7",
    ):
        assert value in record.text
    assert record.text.index("INITIAL-PAGE") < record.text.index("FINAL-PAGE")


def test_references_and_evaluation_annotations_stay_out_of_requests_and_memories(dataset):
    episodes = list(LongMemEvalV2Benchmark(data_root=dataset).load())
    questions = {question["id"]: question for question in read_jsonl(dataset / "questions.jsonl")}
    requests = [case.request for episode in episodes for case in episode.cases]
    assert len({request.id for request in requests}) == len(questions)

    for episode in episodes:
        for record in episode.records:
            assert "PRIVATE-" not in record.text + json.dumps(record.metadata)
        for case in episode.cases:
            original = questions[case.metadata["id"]]
            request = case.request
            assert case.id == original["id"]
            UUID(request.id)
            assert request.id not in questions
            assert request.metadata == {}
            assert request.question == original["question"]
            assert case.reference == original["answer"]
            for key in ("domain", "environment", "question_type", "eval_function"):
                assert case.metadata[key] == original[key]
            expected_images = (
                (str((dataset / original["image"]).resolve()),) if original["image"] else ()
            )
            assert request.attachments == expected_images
            assert "PRIVATE-" not in request.question + json.dumps(request.metadata)


def test_tier_domain_and_limit_select_before_grouping(dataset):
    enterprise = list(
        LongMemEvalV2Benchmark(data_root=str(dataset), domain="enterprise", limit=1).load()
    )
    assert len(enterprise) == 1
    assert enterprise[0].cases[0].metadata["id"] == "q-enterprise"
    assert [record.id for record in enterprise[0].records] == ["enterprise-a"]

    medium = list(LongMemEvalV2Benchmark(data_root=dataset, tier="medium", limit=1).load())
    assert [record.id for record in medium[0].records] == ["web-b"]

    first_two = list(LongMemEvalV2Benchmark(data_root=dataset, limit=2).load())
    assert [case.metadata["id"] for episode in first_two for case in episode.cases] == [
        "q-web-first", "q-web-reversed"
    ]


@pytest.mark.parametrize(
    "options,field",
    [
        ({"tier": "large"}, "tier"),
        ({"domain": "unknown"}, "domain"),
        ({"limit": 0}, "limit"),
        ({"limit": -1}, "limit"),
        ({"limit": True}, "limit"),
        ({"limit": 1.5}, "limit"),
        ({"limit": "1"}, "limit"),
    ],
)
def test_rejects_invalid_filters(dataset, options, field):
    with pytest.raises((TypeError, ValueError), match=field):
        list(LongMemEvalV2Benchmark(data_root=dataset, **options).load())


@pytest.mark.parametrize("filename", ["questions.jsonl", "trajectories.jsonl"])
def test_rejects_duplicate_record_ids(dataset, filename):
    path = dataset / filename
    rows = read_jsonl(path)
    rows.append(deepcopy(rows[0]))
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="(?i)duplicate"):
        list(LongMemEvalV2Benchmark(data_root=dataset).load())


@pytest.mark.parametrize(
    "filename,field,value,match",
    [
        ("questions.jsonl", "id", None, "id"),
        ("questions.jsonl", "question", 42, "question"),
        ("questions.jsonl", "domain", "unknown", "domain"),
        ("trajectories.jsonl", "states", {}, "states"),
        ("trajectories.jsonl", "outcome", "pending", "outcome"),
    ],
)
def test_rejects_malformed_records(dataset, filename, field, value, match):
    path = dataset / filename
    rows = read_jsonl(path)
    if value is None:
        del rows[0][field]
    else:
        rows[0][field] = value
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match=match):
        list(LongMemEvalV2Benchmark(data_root=dataset).load())


def test_rejects_malformed_json_with_file_and_line(dataset):
    path = dataset / "trajectories.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write("{broken json}\n")
    with pytest.raises(ValueError) as caught:
        list(LongMemEvalV2Benchmark(data_root=dataset).load())
    assert "trajectories.jsonl" in str(caught.value)
    assert "5" in str(caught.value)


@pytest.mark.parametrize("field,value", [("state_index", -1), ("action", {"click": "catalog"})])
def test_malformed_state_reports_trajectory_and_field(dataset, field, value):
    path = dataset / "trajectories.jsonl"
    rows = read_jsonl(path)
    rows[0]["states"][0][field] = value
    write_jsonl(path, rows)
    with pytest.raises(ValueError) as caught:
        list(LongMemEvalV2Benchmark(data_root=dataset).load())
    assert "web-b" in str(caught.value)
    assert field in str(caught.value)


def test_rejects_duplicate_question_key_in_haystack_file(dataset):
    path = dataset / "haystacks/lme_v2_small.json"
    path.write_text('{"q-web-first": ["web-a"], "q-web-first": ["web-b"]}')
    with pytest.raises(ValueError, match="(?i)duplicate") as caught:
        list(LongMemEvalV2Benchmark(data_root=dataset).load())
    assert "q-web-first" in str(caught.value)


@pytest.mark.parametrize(
    "replacement,match",
    [
        (None, "q-web-first"),
        (["missing-trajectory"], "missing-trajectory"),
        (["enterprise-a"], "domain"),
        (["web-a", "web-a"], "(?i)duplicate"),
        ([], "q-web-first"),
        ("web-a", "q-web-first"),
    ],
)
def test_rejects_missing_or_invalid_haystack_references(dataset, replacement, match):
    path = dataset / "haystacks/lme_v2_small.json"
    haystacks = json.loads(path.read_text())
    if replacement is None:
        del haystacks["q-web-first"]
    else:
        haystacks["q-web-first"] = replacement
    path.write_text(json.dumps(haystacks))
    with pytest.raises(ValueError, match=match):
        list(LongMemEvalV2Benchmark(data_root=dataset).load())


@pytest.mark.parametrize("relative_path", ["question_screenshots/original.png", "screenshots/web-a/0.png"])
def test_missing_images_report_the_path_and_preparation_hint(dataset, relative_path):
    (dataset / relative_path).unlink()
    with pytest.raises((FileNotFoundError, ValueError)) as caught:
        list(LongMemEvalV2Benchmark(data_root=dataset).load())
    message = str(caught.value)
    assert "question_screenshots" in message if relative_path.startswith("question") else "web-a" in message
    assert "prepare_data.py" in message


def test_domain_filter_only_requires_images_for_selected_questions_and_trajectories(dataset):
    (dataset / "question_screenshots/original.png").unlink()
    (dataset / "screenshots/web-a/0.png").unlink()
    episodes = list(LongMemEvalV2Benchmark(data_root=dataset, domain="enterprise").load())
    assert len(episodes) == 1
    assert episodes[0].cases[0].metadata["id"] == "q-enterprise"


@pytest.mark.parametrize(
    "filename", ["questions.jsonl", "trajectories.jsonl", "haystacks/lme_v2_small.json"]
)
def test_missing_dataset_file_reports_its_path(dataset, filename):
    (dataset / filename).unlink()
    with pytest.raises(FileNotFoundError) as caught:
        list(LongMemEvalV2Benchmark(data_root=dataset).load())
    assert filename in str(caught.value)


def test_empty_domain_selection_fails_instead_of_reporting_an_empty_benchmark(dataset):
    path = dataset / "questions.jsonl"
    write_jsonl(path, [question for question in read_jsonl(path) if question["domain"] == "web"])
    path = dataset / "haystacks/lme_v2_small.json"
    haystacks = json.loads(path.read_text())
    del haystacks["q-enterprise"]
    path.write_text(json.dumps(haystacks))
    with pytest.raises(ValueError, match="(?i)no questions"):
        list(LongMemEvalV2Benchmark(data_root=dataset, domain="enterprise").load())


def test_scoring_rejects_unknown_evaluators_and_empty_aggregation(dataset):
    benchmark = LongMemEvalV2Benchmark(data_root=dataset, limit=1)
    assert benchmark.supports_scoring
    case = list(benchmark.load())[0].cases[0]
    with pytest.raises(ValueError, match="Unsupported.*evaluator"):
        benchmark.score(case, Prediction(case.reference))
    with pytest.raises(RuntimeError, match="No records"):
        benchmark.aggregate([])


def test_registry_constructs_loader_and_stable_ids_survive_reloads(dataset):
    benchmark = instantiate(
        PluginSpec("lme", "longmemeval_v2", {"data_root": dataset, "limit": 1}),
        BENCHMARKS, BaseBenchmark,
    )
    first = next(iter(benchmark.load())).cases[0]
    second = next(iter(benchmark.load())).cases[0]
    assert first.id == second.id == "q-web-first"
    assert first.request.id != second.request.id
    assert first.request.id != first.id


def test_runner_exports_stable_case_id_without_exposing_it_to_model(
    dataset, config_factory, monkeypatch
):
    # A test-only evaluator exercises the output path independently of scoring.
    class FixtureScoringBenchmark(LongMemEvalV2Benchmark):
        supports_scoring = True

        def validate_run(self, agent, generation):
            pass

        def score(self, case, prediction):
            return {"fixture": 1.0}

        def aggregate(self, scores):
            return {"fixture": 1.0}

    monkeypatch.setitem(BENCHMARKS, "fixture_lme", FixtureScoringBenchmark)
    from memory_bench.generators import GENERATORS, BaseGenerator

    class RecordingGenerator(BaseGenerator):
        def generate(self, messages, *, settings, model_adapter=None, attachments=()):
            prompt = json.dumps([message.content for message in messages], default=str)
            assert "q-web-first" not in prompt
            assert "PRIVATE-" not in prompt
            return Prediction("test answer")

        def close(self):
            pass

    monkeypatch.setitem(GENERATORS, "recording", RecordingGenerator)
    config = config_factory(
        benchmarks=[("lme", "fixture_lme", {"data_root": str(dataset), "limit": 1})],
        generation="recording",
        agent_type="fixture_agent",
    )
    output = run(config)
    row = json.loads((output / "predictions.jsonl").read_text())
    assert row["case_id"] == "q-web-first"
    UUID(row["request_id"])
    assert row["request_id"] != row["case_id"]

