import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from memory_bench.agents.ripple_edit import RippleEditAgent
from memory_bench.benchmarks.ripple_edit import CATEGORIES, SUBSETS, RippleEditBenchmark
from memory_bench.config import ComponentSpec, GenerationBinding, PluginSpec, RunConfig, load_config
from memory_bench.generators.base import BaseGenerator
from memory_bench.memories.no_memory import NoMemory
from memory_bench.memories.verbatim import VerbatimMemory
from memory_bench.runner import run
from memory_bench.types import MemoryAugmentation, ModelAdapterRef, Prediction, Request


ROOT = Path(__file__).resolve().parents[1]
def query(*entities):
    return {"prompt": "The country is", "answers": [
        {"value": value, "aliases": aliases} for value, aliases in entities
    ]}


def test_group(*queries, condition="OR"):
    return {"test_queries": list(queries), "test_condition": condition,
            "condition_queries": [query(("PRIVATE CONDITION", []))]}


# Fixture constructor, not a test.
test_group.__test__ = False


def entry(subset="recent"):
    return {"example_type": subset,
            "edit": {"prompt": "The country is France.",
                     "original_fact": {"prompt": "PRIVATE ORIGINAL FACT"}},
            **{category: [test_group(query(("France", ["PRIVATE ALIAS"])))]
               for category in CATEGORIES}}


def write_dataset(tmp_path, entries=None):
    for subset in SUBSETS:
        rows = copy.deepcopy(entries) if entries is not None else [entry(subset), entry(subset)]
        for row in rows:
            row["example_type"] = subset
        (tmp_path / f"{subset}.json").write_text(json.dumps(rows), encoding="utf-8")
    return tmp_path


def load_one(tmp_path, queries, *, condition="OR"):
    row = entry()
    row.update({category: [] for category in CATEGORIES})
    row[CATEGORIES[0]] = [test_group(*queries, condition=condition)]
    benchmark = RippleEditBenchmark(data_root=write_dataset(tmp_path, [row]), subsets=["recent"])
    return benchmark, list(benchmark.load())[0]


def test_loading_subsets_limits_ids_and_reference_privacy(tmp_path):
    root = write_dataset(tmp_path)
    benchmark = RippleEditBenchmark(data_root=root, subsets=["popular", "recent"], limit=1)
    episodes = list(benchmark.load())
    assert [e.id for e in episodes] == ["ripple_edit/recent/0", "ripple_edit/popular/0"]
    full = list(RippleEditBenchmark(data_root=root).load())
    assert len(full) == 6
    assert episodes[1] == full[4]
    for episode in episodes:
        assert len(episode.records) == 1
        assert episode.records[0].text == "The country is France."
        assert episode.records[0].metadata == {}
        assert len(episode.cases) == 6
        for index, case in enumerate(episode.cases):
            assert case.request == Request(id=f"q{index}", question="The country is")
            assert case.metadata["category_index"] == index
            assert CATEGORIES[index] in case.id
            assert case.reference == (("france", "private alias"),)
            assert "PRIVATE" not in repr(episode.records) + repr(case.request)


@pytest.mark.parametrize("options", [
    {"subsets": "recent"}, {"subsets": []}, {"subsets": ["other"]},
    {"subsets": ["recent", "recent"]}, {"subsets": None},
    {"limit": 0}, {"limit": -1}, {"limit": True}, {"limit": 1.5},
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        RippleEditBenchmark(**options)


@pytest.mark.parametrize("entities,answer,expected", [
    ([("France", ["French Republic"])], "The FRENCH republic.", 1),
    ([("Café", [])], "ＣＡＦＥ\u0301", 1),
    ([("New York", [])], "new\n  YORK", 1),
    ([("US", [])], "Russia", 0),
    ([("42", [])], "142", 0),
    ([("US", [])], "(us)", 1),
    ([("A+B", [])], "A+B", 1),
    ([("France", []), ("Germany", ["Deutschland"])], "France and Deutschland", 1),
    ([("France", []), ("Germany", [])], "France", 0),
    ([("", ["France"])], "France", 1),
    ([("France", ["", "   "])], "", 0),
])
def test_matching(tmp_path, entities, answer, expected):
    benchmark, episode = load_one(tmp_path, [query(*entities)])
    result = benchmark.score(episode.cases[0], Prediction(answer))
    assert result["query_accuracy"] == expected
    assert benchmark.aggregate([result])["test_accuracy"] == expected
    assert benchmark.evaluation_details()["protocol"] == "memory_adaptation"


def test_grouping_exclusions_and_denominators(tmp_path):
    row = entry()
    row.update({category: [] for category in CATEGORIES})
    correct = query(("France", []))
    incorrect = query(("Germany", []))
    empty_entity = query(("France", []), ("  ", [""]))
    row[CATEGORIES[0]] = [
        test_group(correct, incorrect),  # OR passes; two query scores.
        test_group(correct, incorrect, condition="AND"),  # AND fails.
        test_group(correct, query(), empty_entity, condition="AND"),  # Ignore exclusions.
        test_group(query()),  # Entire test excluded.
        test_group(),  # Empty test excluded, without adding a phantom query.
    ]
    row[CATEGORIES[1]] = [test_group(incorrect)]
    benchmark = RippleEditBenchmark(data_root=write_dataset(tmp_path, [row]), subsets=["recent"])
    episode = list(benchmark.load())[0]
    assert len(episode.cases) == 6
    assert [c.metadata["query_index"] for c in episode.cases[:2]] == [0, 1]
    scores = [benchmark.score(case, Prediction("France")) for case in episode.cases]
    metrics = benchmark.aggregate(list(reversed(scores)))
    assert metrics["query_accuracy"] == 0.5
    assert metrics["test_accuracy"] == 0.5
    assert metrics["evaluated_queries"] == 6
    assert metrics["evaluated_tests"] == 4
    assert metrics["excluded_queries"] == 3
    assert metrics["excluded_tests"] == 2
    assert metrics[f"category.{CATEGORIES[0]}.test_accuracy"] == pytest.approx(2 / 3)
    assert metrics[f"category.{CATEGORIES[2]}.test_accuracy"] == 0
    assert metrics[f"category.{CATEGORIES[2]}.evaluated_tests"] == 0
    assert metrics["subset.recent.excluded_queries"] == 3
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.aggregate(scores + [scores[0]])
    with pytest.raises(ValueError, match="Incomplete"):
        benchmark.aggregate(scores[:-1])
    # Loading again resets inventory instead of double-counting exclusions.
    list(benchmark.load())
    assert benchmark.aggregate(scores) == metrics


def test_all_unscorable_and_stable_source_query_ids(tmp_path):
    benchmark, episode = load_one(tmp_path, [query(), query(("France", []))])
    assert episode.cases[0].id.endswith("/0/1")
    assert episode.cases[0].metadata["query_index"] == 1
    benchmark, episode = load_one(tmp_path, [query(), query(("", []))])
    assert not episode.cases
    metrics = benchmark.aggregate([])
    assert metrics["query_accuracy"] == metrics["test_accuracy"] == 0
    assert metrics["evaluated_queries"] == metrics["evaluated_tests"] == 0
    assert metrics["excluded_queries"] == 2
    assert metrics["excluded_tests"] == 1


@pytest.mark.parametrize("mutation,location", [
    (lambda e: e.pop(CATEGORIES[0]), "Relation_Specificity"),
    (lambda e: e["edit"].update(prompt=" "), "edit.prompt"),
    (lambda e: e.update(example_type="random"), "example_type"),
    (lambda e: e[CATEGORIES[0]][0].update(test_condition="X"), "test[0].test_condition"),
    (lambda e: e[CATEGORIES[0]][0].update(condition_queries=None), "test[0].condition_queries"),
    (lambda e: e[CATEGORIES[0]][0]["test_queries"][0].update(prompt=None), "query[0].prompt"),
    (lambda e: e[CATEGORIES[0]][0]["test_queries"][0].update(answers=None), "query[0].answers"),
    (lambda e: e[CATEGORIES[0]][0]["test_queries"][0]["answers"][0].update(aliases=[1]),
     "query[0].answers[0].aliases"),
])
def test_malformed_input_has_source_location(tmp_path, mutation, location):
    row = entry()
    mutation(row)
    (tmp_path / "recent.json").write_text(json.dumps([row]))
    with pytest.raises(ValueError) as error:
        list(RippleEditBenchmark(data_root=tmp_path, subsets=["recent"]).load())
    assert "recent.json: edit[0]" in str(error.value)
    assert location in str(error.value)


def test_missing_and_invalid_json(tmp_path):
    benchmark = RippleEditBenchmark(data_root=tmp_path, subsets=["recent"])
    with pytest.raises(ValueError, match="recent.json"):
        list(benchmark.load())
    (tmp_path / "recent.json").write_text("not JSON")
    with pytest.raises(ValueError, match="recent.json"):
        list(benchmark.load())


class RecordingGenerator(BaseGenerator):
    calls = []

    def generate(self, messages, *, settings, model_adapter=None, attachments=()):
        self.calls.append((messages, settings, model_adapter))
        assert not attachments
        assert "PRIVATE" not in repr(messages)
        answer = "France" if any("Memory:\n" in m.content for m in messages) else "Germany"
        return Prediction(answer, {"backend_marker": True})

    def close(self):
        pass


def test_agent_context_adapter_and_timings():
    adapter = ModelAdapterRef("lora", "/test/adapter")

    class Memory(NoMemory):
        def augment(self, request):
            return MemoryAugmentation(context="fact one", model_adapter=adapter,
                                      context_items=({"type": "text", "value": "fact two"},))

    generator = RecordingGenerator()
    settings = {"temperature": 0.0}
    result = RippleEditAgent().answer(Request("q", "country?"), Memory(), generator, settings)
    messages, received_settings, received_adapter = generator.calls[-1]
    assert messages[1].content == "Memory:\nfact one\nfact two"
    assert messages[2].content == "Query:\ncountry?"
    assert received_settings == settings
    assert received_adapter == adapter
    assert result.metadata["backend_marker"] is True
    assert all(v >= 0 for v in result.metadata["timings"].values())


def test_agent_rejects_images_before_generation():
    agent = RippleEditAgent()
    with pytest.raises(ValueError, match="attachments"):
        agent.answer(Request("q", "?", attachments=("image.png",)), NoMemory(), None, {})

    class Images(NoMemory):
        def augment(self, request):
            return MemoryAugmentation(context_items=({"type": "image", "value": "image.png"},))

    with pytest.raises(ValueError, match="images"):
        agent.answer(Request("q", "?"), Images(), None, {})


def test_runner_matrix_and_example_config(tmp_path, monkeypatch):
    from memory_bench.generators import GENERATORS

    monkeypatch.setitem(GENERATORS, "ripple_fixture", RecordingGenerator)
    root = write_dataset(tmp_path)
    config = load_config(ROOT / "configs/ripple-edit.toml")
    assert config.agent.type == config.benchmarks[0].type == "ripple_edit"
    config = RunConfig(
        benchmarks=(ComponentSpec("ripple", "ripple_edit", {"data_root": str(root), "limit": 1}),),
        memories=(ComponentSpec("none", "none"), ComponentSpec("verbatim", "verbatim")),
        agent=ComponentSpec("ripple", "ripple_edit", generation=GenerationBinding("reader")),
        generators={"reader": PluginSpec("reader", "ripple_fixture")},
        results_dir=tmp_path / "results", artifacts_dir=tmp_path / "artifacts", snapshot={},
    )
    result = run(config)
    summary = json.loads((result / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert [p["metrics"]["test_accuracy"] for p in summary["pairs"]] == [0, 1]
    assert all(p["episodes"] == 3 and p["questions"] == 18 for p in summary["pairs"])
    predictions = [json.loads(line) for line in (result / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 36
    assert all(row["evaluation"]["protocol"] == "memory_adaptation" for row in predictions)
    assert len((result / "generations.jsonl").read_text().splitlines()) == len(predictions)


@pytest.mark.parametrize("first", ["agents", "benchmarks", "agents.ripple_edit", "benchmarks.ripple_edit"])
def test_registry_import_order_without_optional_dependencies(first, tmp_path):
    code = f"""
import memory_bench.{first}
from memory_bench.agents import AGENTS
from memory_bench.benchmarks import BENCHMARKS
assert AGENTS['ripple_edit'].__name__ == 'RippleEditAgent'
assert BENCHMARKS['ripple_edit'].__name__ == 'RippleEditBenchmark'
assert AGENTS['ripple_edit'].__module__ == 'memory_bench.agents.ripple_edit'
assert BENCHMARKS['ripple_edit'].__module__ == 'memory_bench.benchmarks.ripple_edit'
"""
    subprocess.run([sys.executable, "-S", "-c", code], cwd=tmp_path, check=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)


def test_standalone_tree_loads_and_runs_without_third_party(tmp_path):
    # The subprocess has only a staged source/data tree and the standard library.
    shutil.copytree(ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "data/ripple_edit", tmp_path / "data/ripple_edit")
    code = """
import json
from pathlib import Path
import memory_bench.benchmarks.ripple_edit as ripple
from memory_bench.config import ComponentSpec, GenerationBinding, PluginSpec, RunConfig
from memory_bench.generators import register_generator
from memory_bench.generators.base import BaseGenerator
from memory_bench.runner import run
from memory_bench.types import Prediction
assert Path(ripple.__file__).is_relative_to(Path.cwd())
assert not Path('third_party').exists()
assert sum(1 for _ in ripple.RippleEditBenchmark().load()) == 4755
@register_generator('fake')
class Fake(BaseGenerator):
    def generate(self, messages, **kwargs):
        return Prediction('photography')
    def close(self):
        pass
config = RunConfig(
    benchmarks=(ComponentSpec('ripple', 'ripple_edit', {'limit': 1}),),
    memories=(ComponentSpec('none', 'none'), ComponentSpec('verbatim', 'verbatim')),
    agent=ComponentSpec('ripple', 'ripple_edit', generation=GenerationBinding('reader')),
    generators={'reader': PluginSpec('reader', 'fake')},
    results_dir=Path('results'), artifacts_dir=Path('artifacts'), snapshot={},
)
output = run(config)
summary = json.loads((output / 'summary.json').read_text())
assert summary['status'] == 'completed'
assert len(summary['pairs']) == 2
assert all(p['questions'] > 0 for p in summary['pairs'])
"""
    completed = subprocess.run([sys.executable, "-S", "-c", code], cwd=tmp_path,
                               env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
