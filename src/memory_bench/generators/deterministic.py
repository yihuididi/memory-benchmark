import re
from collections.abc import Mapping, Sequence
from typing import Any
from memory_bench.generators import register_generator
from memory_bench.generators.base import BaseGenerator, UnsupportedModelAdapterError
from memory_bench.types import Message, ModelAdapterRef, Prediction


@register_generator("deterministic")
class DeterministicGenerator(BaseGenerator):
    """Read synthetic facts only from the shared agent's memory context."""

    def __init__(self, *, unknown_answer: str = "UNKNOWN") -> None:
        if not isinstance(unknown_answer, str):
            raise ValueError("unknown_answer must be a string")
        self.unknown_answer = unknown_answer

    def generate(
        self,
        messages: Sequence[Message],
        *,
        settings: Mapping[str, Any],
        model_adapter: ModelAdapterRef | None = None,
        attachments: Sequence[str] = (),
    ) -> Prediction:
        if model_adapter is not None:
            raise UnsupportedModelAdapterError(
                f"DeterministicGenerator does not support model adapters "
                f"(requested kind={model_adapter.kind!r}, "
                f"reference={model_adapter.reference!r}). Configure a generation "
                "backend that supports this adapter kind."
            )
        if attachments:
            raise ValueError(
                "DeterministicGenerator does not support attachments. Configure "
                "a generation backend that can read the request's attachments."
            )

        context = ""
        question = ""
        for message in messages:
            if message.role != "user":
                continue
            if message.content.startswith("Memory context:\n"):
                context = message.content.removeprefix("Memory context:\n")
            elif message.content.startswith("Question:\n"):
                question = message.content.removeprefix("Question:\n")

        match = re.fullmatch(r"What is the value of ([\w-]+)\?", question)
        if match:
            requested_key = match.group(1)
            for line in context.splitlines():
                key, delimiter, value = line.partition(":")
                if delimiter and key.strip() == requested_key:
                    return Prediction(answer=value.strip())
        return Prediction(answer=self.unknown_answer)

    def close(self) -> None:
        pass
