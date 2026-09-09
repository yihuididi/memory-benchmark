from collections.abc import Iterable, Mapping, Sequence
from memory_bench.benchmarks import register_benchmark
from memory_bench.benchmarks.base import BaseBenchmark
from memory_bench.types import Episode, EvaluationCase, MemoryRecord, Prediction, Request


@register_benchmark("synthetic")
class SyntheticBenchmark(BaseBenchmark):
    """Two episodes with repeated keys and different answers for isolation."""

    def __init__(self, *, prefix: str = "demo") -> None:
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a nonempty string")
        self.prefix = prefix

    def load(self) -> Iterable[Episode]:
        for number, values in enumerate(
            (("cobalt", "orchid"), ("amber", "cedar")), start=1
        ):
            facts = dict(zip(("favorite_color", "secret_word"), values, strict=True))
            episode_id = f"{self.prefix}-{number}"
            yield Episode(
                id=episode_id,
                records=tuple(
                    MemoryRecord(id=key, text=f"{key}: {value}")
                    for key, value in facts.items()
                ),
                cases=tuple(
                    EvaluationCase(
                        request=Request(
                            id=key, question=f"What is the value of {key}?"
                        ),
                        reference=value,
                    )
                    for key, value in facts.items()
                ),
            )

    def score(
        self, case: EvaluationCase, prediction: Prediction
    ) -> dict[str, float]:
        matches = (
            prediction.answer.strip().casefold()
            == str(case.reference).strip().casefold()
        )
        return {"exact_match": float(matches)}

    def aggregate(self, scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
        return {
            "exact_match": (
                sum(score["exact_match"] for score in scores) / len(scores)
                if scores
                else 0.0
            )
        }
