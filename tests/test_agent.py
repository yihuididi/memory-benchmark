from __future__ import annotations

import pytest

from memory_bench.agents.shared import SharedAgent
from memory_bench.generators.deterministic import DeterministicGenerator
from memory_bench.types import (
    MemoryAugmentation,
    Message,
    ModelAdapterRef,
    Prediction,
    Request,
)


def test_shared_agent_forwards_memory_binding_without_carrying_prompt_history(tmp_path):
    adapter = ModelAdapterRef("lora", str(tmp_path / "adapter"), {"rank": 8})
    attachments = (str(tmp_path / "question.txt"),)
    settings = {"temperature": 0.0, "max_tokens": 64}
    calls = []
    requests_seen = []
    prediction = Prediction("generated answer", {"tokens": 3})

    class Memory:
        def augment(self, request):
            requests_seen.append(request)
            return MemoryAugmentation(f"CONTEXT-{request.id}", adapter)

    class Generation:
        def generate(self, messages, *, settings, model_adapter=None, attachments=()):
            calls.append((tuple(messages), settings, model_adapter, attachments))
            return prediction

    agent = SharedAgent()
    first = Request("first", "FIRST-QUESTION", attachments=attachments)
    second = Request("second", "SECOND-QUESTION")
    memory, generation = Memory(), Generation()

    first_result = agent.answer(first, memory, generation, settings)
    second_result = agent.answer(second, memory, generation, settings)
    for result in (first_result, second_result):
        assert result.answer == prediction.answer
        assert result.metadata["tokens"] == 3
        assert result.metadata["timings"]["memory_query_duration_seconds"] >= 0
        assert result.metadata["timings"]["reader_duration_seconds"] >= 0
    assert prediction.metadata == {"tokens": 3}

    assert requests_seen == [first, second]
    first_messages, forwarded_settings, forwarded_adapter, forwarded_files = calls[0]
    assert forwarded_settings is settings
    assert forwarded_adapter is adapter
    assert forwarded_files == attachments
    first_prompt = "\n".join(message.content for message in first_messages)
    second_prompt = "\n".join(message.content for message in calls[1][0])
    assert "CONTEXT-first" in first_prompt
    assert "FIRST-QUESTION" in first_prompt
    assert "CONTEXT-second" in second_prompt
    assert "SECOND-QUESTION" in second_prompt
    assert "CONTEXT-first" not in second_prompt
    assert "FIRST-QUESTION" not in second_prompt
    assert calls[1][3] == ()


def test_offline_generator_explicitly_rejects_unsupported_model_adapters():
    generation = DeterministicGenerator()
    try:
        with pytest.raises((ValueError, NotImplementedError), match="(?i)adapter"):
            generation.generate(
                (Message("user", "question"),),
                settings={},
                model_adapter=ModelAdapterRef("lora", "artifacts/trained-adapter"),
            )
    finally:
        generation.close()
