"""Data passed between benchmarks, memory backends, and the shared agent.

Evaluation references live in ``EvaluationCase`` only. Memory and generation
backends receive ``Request`` objects or messages, never evaluation cases.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Request:
    id: str
    question: str
    attachments: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    request: Request
    reference: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    # Stable source ID for results/evaluation only; never passed to the agent.
    id: str | None = None


@dataclass(frozen=True, slots=True)
class Episode:
    id: str
    records: tuple[MemoryRecord, ...]
    cases: tuple[EvaluationCase, ...]


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str | tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ModelAdapterRef:
    """A declarative adapter location; backends decide how to load it.

    For example, ``kind="lora"`` and ``reference=str(workspace / "adapter")``.
    Options hold backend-specific loading settings, not model objects.
    """

    kind: str
    reference: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryAugmentation:
    context: str = ""
    model_adapter: ModelAdapterRef | None = None
    # Ordered evidence, using upstream {type: text|image, value: ...} items.
    context_items: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Prediction:
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
