"""Contract for dataset loading and benchmark-owned evaluation."""

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from memory_bench.config import PluginSpec

from memory_bench.generators.base import BaseGenerator
from memory_bench.types import Episode, EvaluationCase, Prediction


class BaseBenchmark(ABC):
    supports_scoring: bool = True

    generation: BaseGenerator | None = None
    generation_settings: dict[str, Any] | None = None

    def bind_generation(self, generator: BaseGenerator, settings: Mapping[str, Any]) -> None:
        """Borrow a runner-owned generator with independent inference settings."""
        self.generation = generator
        self.generation_settings = copy.deepcopy(dict(settings))

    @abstractmethod
    def load(self) -> Iterable[Episode]:
        """Load episodes, keeping evaluation annotations out of public inputs."""

    @abstractmethod
    def score(self, case: EvaluationCase, prediction: Prediction) -> dict[str, float]:
        """Score one answer using evaluator-only references and annotations."""

    @abstractmethod
    def aggregate(self, scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
        """Compute this benchmark's metrics for a complete experiment pair."""

    def validate_run(self, agent: PluginSpec, generation: PluginSpec) -> None:
        """Validate configuration before preparing memory or running inference."""

    def evaluation_details(self) -> dict[str, Any]:
        """Return diagnostics for the most recently scored prediction."""
        return {}

    def close(self) -> None:
        """Release evaluator resources, if any."""
