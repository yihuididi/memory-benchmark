"""Contract for model invocation, including optional trained adapters."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from memory_bench.types import Message, ModelAdapterRef, Prediction


class BaseGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        settings: Mapping[str, Any],
        model_adapter: ModelAdapterRef | None = None,
        attachments: Sequence[str] = (),
    ) -> Prediction:
        """Generate independently; reject unsupported adapters explicitly.

        Instances start without an active adapter. ``model_adapter=None`` means
        the base model, even when an earlier call used an adapter.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources, including any loaded model adapter."""


class UnsupportedModelAdapterError(ValueError):
    """The selected generator cannot honor the requested model adapter."""
