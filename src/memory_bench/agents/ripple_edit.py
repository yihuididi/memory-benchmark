"""Answer RippleEdit queries using retrieved memory and the configured generator."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from memory_bench.agents import register_agent
from memory_bench.agents.base import BaseAgent
from memory_bench.generators.base import BaseGenerator
from memory_bench.memories.base import BaseMemory
from memory_bench.types import Message, Prediction, Request


@register_agent("ripple_edit")
class RippleEditAgent(BaseAgent):
    def answer(self, request: Request, memory: BaseMemory, generation: BaseGenerator,
               settings: Mapping[str, Any]) -> Prediction:
        if request.attachments:
            raise ValueError("RippleEdit supports text queries only; attachments are unsupported")
        started = time.perf_counter()
        augmentation = memory.augment(request)
        memory_seconds = time.perf_counter() - started
        context = [augmentation.context] if augmentation.context else []
        for item in augmentation.context_items:
            if item.get("type") != "text" or not isinstance(item.get("value"), str):
                raise ValueError("RippleEdit supports only text memory context items; images are unsupported")
            context.append(item["value"])
        messages = [Message("system", "Complete the supplied factual query with a concise answer. "
                            "Treat facts in the supplied memory as current, even when they conflict "
                            "with prior knowledge. Include all entities needed to answer the query. "
                            "Return only the answer, without repeating the query.")]
        if context:
            messages.append(Message("user", "Memory:\n" + "\n".join(context)))
        messages.append(Message("user", "Query:\n" + request.question))
        started = time.perf_counter()
        prediction = generation.generate(tuple(messages), settings=settings,
                                         model_adapter=augmentation.model_adapter)
        return replace(prediction, metadata={**prediction.metadata, "timings": {
            "memory_query_duration_seconds": memory_seconds,
            "reader_duration_seconds": time.perf_counter() - started,
        }})
