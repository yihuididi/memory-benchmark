"""Optional repository agent with upstream prompts and evidence budget.

Generation still goes through BaseGenerator; no upstream reader is instantiated.
Custom agents, including SharedAgent, can also use the benchmark's scorer.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from memory_bench.benchmarks.longmemeval_v2 import harness
from memory_bench.agents import register_agent
from memory_bench.agents.base import BaseAgent
from memory_bench.generators.base import BaseGenerator
from memory_bench.memories.base import BaseMemory
from memory_bench.types import Message, Prediction, Request


@register_agent("longmemeval_v2")
class LongMemEvalV2Agent(BaseAgent):
    def __init__(self, *, domain: str, memory_context_max_tokens: int = 200000,
                 processor_path: str = "Qwen/Qwen3.5-9B", processor_revision: str | None = None,
                 local_files_only: bool = True) -> None:
        if domain not in harness.DOMAIN_SYSTEM_PROMPTS:
            raise ValueError("domain must be web or enterprise")
        if type(memory_context_max_tokens) is not int or memory_context_max_tokens <= 0:
            raise ValueError("memory_context_max_tokens must be a positive integer")
        self.domain, self.memory_context_max_tokens = domain, memory_context_max_tokens
        self.processor_path, self.processor_revision = processor_path, processor_revision
        self.local_files_only = local_files_only
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            try:
                from transformers import AutoProcessor
            except ImportError as exc:
                raise RuntimeError("Evidence token counting requires: uv sync --extra transformers-vision") from exc
            self._processor = AutoProcessor.from_pretrained(
                self.processor_path, revision=self.processor_revision,
                local_files_only=self.local_files_only, trust_remote_code=False,
            )
        return self._processor

    def answer(self, request: Request, memory: BaseMemory, generation: BaseGenerator,
               settings: Mapping[str, Any]) -> Prediction:
        if len(request.attachments) > 1:
            raise ValueError("LongMemEval-V2 questions support one question screenshot")
        started = time.perf_counter()
        augmentation = memory.augment(request)
        memory_seconds = time.perf_counter() - started
        context = ([{"type": "text", "value": augmentation.context}] if augmentation.context.strip() else [])
        context.extend(augmentation.context_items)
        context = harness.validate_memory_context_items(context, question_id=request.id)
        original = kept = 0
        if context:
            context, original, kept = harness.truncate_memory_context(
                context, max_tokens=self.memory_context_max_tokens,
                question_id=request.id, processor=self._get_processor(),
            )
        _, log_messages = harness.build_messages(
            harness.DOMAIN_SYSTEM_PROMPTS[self.domain], request.question,
            request.attachments[0] if request.attachments else None, context,
        )
        messages = tuple(Message(m["role"], tuple(m["content"]) if isinstance(m["content"], list)
                                 else m["content"]) for m in log_messages)
        started = time.perf_counter()
        prediction = generation.generate(messages, settings=settings, model_adapter=augmentation.model_adapter)
        details = {
            "memory_query_duration_seconds": memory_seconds,
            "reader_duration_seconds": time.perf_counter() - started,
            "memory_context_original_token_count": original, "memory_context_token_count": kept,
            "memory_context_truncated": original > kept, "prompt_messages": log_messages,
            "processor_path": self.processor_path, "processor_revision": self.processor_revision,
            "memory_context_max_tokens": self.memory_context_max_tokens,
        }
        return replace(prediction, metadata={**prediction.metadata, "longmemeval_v2": details,
                                            "timings": {key: details[key] for key in
                                                        ("memory_query_duration_seconds", "reader_duration_seconds")}})
