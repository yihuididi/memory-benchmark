"""RippleEdits memory adaptation, independent of the upstream runtime.

Source: edenbiran/RippleEdits, revision 54f3b88af4895a3aacb580ec63ce7ae857185040.
See NOTICE.md and THIRD_PARTY_LICENSES/RIPPLE_EDITS_LICENSE.txt.
Preconditions and edit-success gating are deliberately not part of this protocol.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from memory_bench.benchmarks import register_benchmark
from memory_bench.benchmarks.base import BaseBenchmark
from memory_bench.types import Episode, EvaluationCase, MemoryRecord, Prediction, Request


SUBSETS = ("recent", "random", "popular")
CATEGORIES = (
    "Relation_Specificity", "Logical_Generalization", "Subject_Aliasing",
    "Compositionality_I", "Compositionality_II", "Forgetfulness",
)
_GROUP_FIELDS = ("subset_index", "edit_index", "category_index", "test_index")


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _require(value: Any, kind: type, location: str) -> Any:
    if not isinstance(value, kind):
        raise ValueError(f"{location}: expected {kind.__name__}")
    return value


def _text(value: Any, location: str) -> str:
    value = _require(value, str, location)
    if not value.strip():
        raise ValueError(f"{location}: expected nonempty text")
    return value


def _query(query: Any, location: str) -> tuple[str, tuple[tuple[str, ...], ...]]:
    query = _require(query, dict, location)
    prompt = _text(query.get("prompt"), f"{location}.prompt")
    answers = _require(query.get("answers"), list, f"{location}.answers")
    entities = []
    for index, answer in enumerate(answers):
        where = f"{location}.answers[{index}]"
        answer = _require(answer, dict, where)
        value = _require(answer.get("value"), str, f"{where}.value")
        aliases = _require(answer.get("aliases"), list, f"{where}.aliases")
        for alias in aliases:
            _require(alias, str, f"{where}.aliases")
        entities.append(tuple(dict.fromkeys(
            normalized for candidate in (value, *aliases)
            if (normalized := _normalize(candidate))
        )))
    # An entity with no names cannot be dropped: all entities are required.
    return prompt, tuple(entities) if entities and all(entities) else ()


@register_benchmark("ripple_edit")
class RippleEditBenchmark(BaseBenchmark):
    def __init__(self, *, data_root: str | Path = "data/ripple_edit",
                 subsets: Sequence[str] = SUBSETS, limit: int | None = None) -> None:
        if (isinstance(subsets, (str, bytes)) or not isinstance(subsets, Sequence)
                or not subsets or any(s not in SUBSETS for s in subsets)
                or len(set(subsets)) != len(subsets)):
            raise ValueError(f"subsets must be a nonempty, unique sequence from {SUBSETS}")
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("limit must be a positive integer or None")
        self.data_root = Path(data_root)
        self.subsets = tuple(s for s in SUBSETS if s in subsets)
        self.limit = limit
        # Loaded test inventory supplies exclusions even when no requests survive.
        self._tests: dict[tuple[int, ...], tuple[str, int, int]] = {}
        self._details: dict[str, Any] = {}

    def load(self) -> Iterable[Episode]:
        self._tests.clear()
        self._details = {}
        for subset in self.subsets:
            path = self.data_root / f"{subset}.json"
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"{path}: could not read RippleEdit dataset: {exc}") from exc
            _require(data, list, str(path))
            for edit_index, entry in enumerate(data[:self.limit]):
                location = f"{path}: edit[{edit_index}]"
                entry = _require(entry, dict, location)
                if entry.get("example_type") != subset:
                    raise ValueError(f"{location}.example_type: expected {subset!r}")
                edit = _require(entry.get("edit"), dict, f"{location}.edit")
                fact = _text(edit.get("prompt"), f"{location}.edit.prompt")
                episode_id = f"ripple_edit/{subset}/{edit_index}"
                cases = []
                for category_index, category in enumerate(CATEGORIES):
                    tests = _require(entry.get(category), list, f"{location}.{category}")
                    for test_index, test in enumerate(tests):
                        where = f"{location}.{category}.test[{test_index}]"
                        test = _require(test, dict, where)
                        condition = test.get("test_condition")
                        if condition not in ("OR", "AND"):
                            raise ValueError(f"{where}.test_condition: expected OR or AND")
                        queries = _require(test.get("test_queries"), list, f"{where}.test_queries")
                        conditions = _require(test.get("condition_queries"), list,
                                              f"{where}.condition_queries")
                        for index, query in enumerate(conditions):
                            _query(query, f"{where}.condition_queries[{index}]")
                        key = (SUBSETS.index(subset), edit_index, category_index, test_index)
                        scorable = 0
                        for query_index, query in enumerate(queries):
                            prompt, reference = _query(query, f"{where}.query[{query_index}]")
                            if not reference:
                                continue
                            scorable += 1
                            case_id = f"{episode_id}/{category}/{test_index}/{query_index}"
                            cases.append(EvaluationCase(
                                # Public IDs are opaque within the episode; no category or answers.
                                request=Request(id=f"q{len(cases)}", question=prompt),
                                reference=reference,
                                metadata={**dict(zip(_GROUP_FIELDS, key)),
                                          "query_index": query_index, "condition": condition},
                                id=case_id,
                            ))
                        self._tests[key] = (condition, scorable, len(queries) - scorable)
                yield Episode(id=episode_id,
                              records=(MemoryRecord(id="edit", text=fact),), cases=tuple(cases))

    def score(self, case: EvaluationCase, prediction: Prediction) -> dict[str, float]:
        answer = _normalize(prediction.answer)
        matched = [any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", answer)
                       is not None for alias in entity) for entity in case.reference]
        correct = bool(matched) and all(matched)
        self._details = {"protocol": "memory_adaptation", "matched_entities": matched,
                         "test_condition": case.metadata["condition"]}
        return {"query_accuracy": float(correct),
                **{field: float(case.metadata[field]) for field in _GROUP_FIELDS},
                "query_index": float(case.metadata["query_index"])}

    def aggregate(self, scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
        grouped: dict[tuple[int, ...], dict[int, float]] = defaultdict(dict)
        for score in scores:
            key = tuple(int(score[field]) for field in _GROUP_FIELDS)
            query_index = int(score["query_index"])
            if key not in self._tests:
                raise ValueError(f"Unknown RippleEdit test group: {key}")
            if query_index in grouped[key]:
                raise ValueError(f"Duplicate RippleEdit query score: {key}, {query_index}")
            grouped[key][query_index] = score["query_accuracy"]

        def summarize(keys: Iterable[tuple[int, ...]]) -> dict[str, float]:
            query_correct = query_count = test_correct = test_count = 0
            excluded_queries = excluded_tests = 0
            for key in keys:
                condition, expected, excluded = self._tests[key]
                values = list(grouped.get(key, {}).values())
                if len(values) != expected:
                    raise ValueError(f"Incomplete RippleEdit test group: {key}")
                excluded_queries += excluded
                if not expected:
                    excluded_tests += 1
                    continue
                query_count += expected
                query_correct += sum(values)
                test_count += 1
                test_correct += int(any(values) if condition == "OR" else all(values))
            return {"query_accuracy": query_correct / query_count if query_count else 0.0,
                    "test_accuracy": test_correct / test_count if test_count else 0.0,
                    "evaluated_queries": float(query_count), "evaluated_tests": float(test_count),
                    "excluded_queries": float(excluded_queries), "excluded_tests": float(excluded_tests)}

        metrics = summarize(self._tests)
        for category_index, category in enumerate(CATEGORIES):
            metrics.update({f"category.{category}.{name}": value for name, value in
                            summarize(k for k in self._tests if k[2] == category_index).items()})
        for subset in self.subsets:
            metrics.update({f"subset.{subset}.{name}": value for name, value in
                            summarize(k for k in self._tests if k[0] == SUBSETS.index(subset)).items()})
        return metrics

    def evaluation_details(self) -> dict[str, Any]:
        return dict(self._details)
