import json
from types import SimpleNamespace

import pytest

from memory_bench.benchmarks.longmemeval_v2 import harness
from memory_bench.agents.longmemeval_v2 import LongMemEvalV2Agent
from memory_bench.types import MemoryAugmentation, Prediction, Request


def test_reader_uses_official_prompt_and_preserves_ordered_images(tmp_path, monkeypatch):
    question_image = tmp_path / "question.png"
    question_image.write_bytes(b"question image")
    evidence_image = tmp_path / "evidence.png"
    evidence_image.write_bytes(b"evidence image")
    items = [{"type": "text", "value": "Evidence before image"},
             {"type": "image", "value": str(evidence_image)},
             {"type": "text", "value": "Evidence after image"}]
    request = Request("invocation-id", "Question text", (str(question_image),))
    memory_calls, model_calls = [], []
    def augment(value):
        memory_calls.append(value)
        assert value.metadata == {}
        return MemoryAugmentation(context_items=tuple(items))
    def generate(messages, **kwargs):
        model_calls.append((messages, kwargs))
        return Prediction(r"\boxed{answer}", {"model": "reader"})
    agent = LongMemEvalV2Agent(domain="web")
    monkeypatch.setattr(agent, "_get_processor", lambda: object())
    monkeypatch.setattr(harness, "truncate_memory_context", lambda context, **kwargs: (context, 10, 10))
    ticks = iter([10., 12., 20., 27.])
    monkeypatch.setattr("memory_bench.agents.longmemeval_v2.time.perf_counter", lambda: next(ticks))
    prediction = agent.answer(request, SimpleNamespace(augment=augment), SimpleNamespace(generate=generate), {"max_tokens": 5})
    messages, kwargs = model_calls[0]
    assert memory_calls == [request]
    _, expected = harness.build_messages(harness.DOMAIN_SYSTEM_PROMPTS["web"], request.question, str(question_image), items)
    assert messages[0].content == expected[0]["content"]
    assert list(messages[1].content) == expected[1]["content"]
    assert "invocation-id" not in json.dumps(expected)
    assert kwargs == {"settings": {"max_tokens": 5}, "model_adapter": None}
    details = prediction.metadata["longmemeval_v2"]
    assert details["memory_query_duration_seconds"] == 2
    assert details["reader_duration_seconds"] == 7
    assert details["memory_context_token_count"] == 10


def test_no_memory_needs_no_transformers_processor(monkeypatch):
    agent = LongMemEvalV2Agent(domain="enterprise")
    monkeypatch.setattr(agent, "_get_processor", lambda: pytest.fail("No evidence needs no tokenization"))
    prediction = agent.answer(Request("id", "q"), SimpleNamespace(augment=lambda _: MemoryAugmentation()),
                              SimpleNamespace(generate=lambda *a, **kw: Prediction("UNKNOWN")), {})
    details = prediction.metadata["longmemeval_v2"]
    assert details["memory_context_token_count"] == 0
    assert details["prompt_messages"][1]["content"][0]["text"] == "### Memory context:\n(empty)"


def test_truncation_keeps_only_whole_prefix_items_and_counts_images(monkeypatch):
    items = [{"type": "text", "value": "first"}, {"type": "image", "value": "image.png"},
             {"type": "text", "value": "last"}]
    monkeypatch.setattr(harness, "load_memory_context_images", lambda x: [None, object(), None])
    seen_images = []
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            parts = messages[0]["content"]
            return str(sum(100 if part["type"] == "image" else 10 for part in parts) + 5)
        def __call__(self, *, text, images, return_tensors):
            seen_images.append(images)
            return {"input_ids": SimpleNamespace(shape=(1, int(text)))}
    kept, original, count = harness.truncate_memory_context(items, max_tokens=115, question_id="id", processor=Processor())
    assert kept == items[:2]
    assert (original, count) == (125, 115)
    assert any(images and len(images) == 1 for images in seen_images)
    kept, original, count = harness.truncate_memory_context(items, max_tokens=14, question_id="id", processor=Processor())
    assert (kept, original, count) == ([], 125, 0)


def test_invalid_evidence_fails_before_reader():
    agent = LongMemEvalV2Agent(domain="web")
    with pytest.raises(RuntimeError, match="invalid type"):
        agent.answer(Request("id", "q"), SimpleNamespace(augment=lambda _: MemoryAugmentation(
            context_items=({"type": "audio", "value": "file.wav"},))),
            SimpleNamespace(generate=lambda *a, **kw: pytest.fail("Bad evidence must fail first")), {})
