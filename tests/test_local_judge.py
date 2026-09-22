from contextlib import nullcontext
from types import SimpleNamespace
import sys

import pytest
from pydantic import ValidationError

from memory_bench.benchmarks.longmemeval_v2 import Judge, binary_judgement_schema
from memory_bench.generators.transformers import TransformersGenerator
from memory_bench.generators.base import UnsupportedModelAdapterError
from memory_bench.types import Message


@pytest.mark.parametrize("raw", [
    '{"label": true, "reason": "boolean is not a label"}',
    '{"label": "1", "reason": "string is not a label"}',
    '{"label": 2, "reason": "not binary"}',
    '{"label": 1.0, "reason": "not an integer"}',
    '{"label": 1}', '{"label": 0, "reason": ""}',
    '{"label": 1, "reason": "ok", "extra": "unrequested"}',
    '```json\n{"label": 1, "reason": "ok"}\n```',
    '{"label": 1, "reason": "unfinished',
])
def test_judge_schema_rejects_invalid_or_incomplete_output(raw):
    with pytest.raises(ValidationError):
        binary_judgement_schema().model_validate_json(raw)



@pytest.fixture
def local_backend(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{}')
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096), dtype="float32")
    model.eval = lambda: model
    tokenizer = SimpleNamespace(chat_template="template", model_max_length=4096,
        apply_chat_template=lambda *a, **kw: "prompt", encode=lambda *a: [1, 2])
    def load(kind, value):
        def from_pretrained(path, **kwargs):
            calls.append((kind, kwargs))
            return value
        return SimpleNamespace(from_pretrained=from_pretrained)
    def generator(wrapped, schema):
        calls.append(("schema", schema))
        def generate(prompt, **kwargs):
            calls.append(("generate", kwargs))
            return '{"label": 1, "reason": "Correct insight"}' if schema else "plain answer"
        return generate
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=load("model", model), AutoTokenizer=load("tokenizer", tokenizer)))
    monkeypatch.setitem(sys.modules, "outlines", SimpleNamespace(
        from_transformers=lambda *a: "wrapped", Generator=generator))
    monkeypatch.setattr("memory_bench.generators.transformers.version", lambda name: "test-version")
    return TransformersGenerator(model_path=str(tmp_path), device="cpu"), calls


def test_local_backend_never_downloads_missing_model(tmp_path):
    generator = TransformersGenerator(model_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Local model missing"):
        generator.preflight()
    generator.close()


def test_local_backend_constrains_json_reuses_model_and_supports_text(local_backend):
    generator, calls = local_backend
    judge = Judge(generator, {"max_new_tokens": 80, "chat_template_kwargs": {"enable_thinking": False}})
    for _ in range(2):
        assert judge.score("llm_gotchas_checker", question="q", reference="private", raw="a", parsed="a")
    assert generator.generate([Message("user", "q")], settings={}).answer == "plain answer"
    assert sum(kind == "model" for kind, _ in calls) == 1
    assert [value for kind, value in calls if kind == "schema"] == [binary_judgement_schema(), binary_judgement_schema(), None]
    for kind, kwargs in calls:
        if kind in ("model", "tokenizer"):
            assert kwargs["local_files_only"] is True
            assert kwargs["trust_remote_code"] is False
    assert judge.details["reason"] == "Correct insight"
    assert judge.details["input_tokens"] == 2
    assert judge.details["versions"]["outlines"] == "test-version"
    judge.close()
    assert generator.model is not None  # Borrowed resource remains owned by runner.
    generator.close()
    assert generator.model is generator.tokenizer is generator.wrapped is None


def test_local_backend_does_not_truncate_overflow(local_backend):
    generator, calls = local_backend
    generator._load()
    generator.tokenizer.encode = lambda *a: [1] * 4000
    with pytest.raises(ValueError, match="never truncated"):
        generator.generate([Message("user", "q")], settings={})
    assert not any(kind == "generate" for kind, _ in calls)


def test_local_backend_invalid_json_is_an_evaluation_failure(local_backend, monkeypatch):
    generator, _ = local_backend
    monkeypatch.setattr(sys.modules["outlines"], "Generator", lambda *a: lambda *a, **kw: '{"label": 1')
    judge = Judge(generator, {})
    with pytest.raises(ValueError, match="Could not parse"):
        judge.score("llm_gotchas_checker", question="q", reference="r", raw="a", parsed="a")
    assert judge.details == {}


@pytest.mark.parametrize("settings", [
    {"max_new_tokens": 0}, {"do_sample": "false"}, {"unknown": 1},
    {"chat_template_kwargs": {"tokenize": True}},
])
def test_local_backend_rejects_invalid_settings_before_loading(tmp_path, settings):
    generator = TransformersGenerator(model_path=str(tmp_path))
    with pytest.raises(ValueError):
        generator.generate([Message("user", "q")], settings=settings)
    assert generator.model is None


def test_local_backend_rejects_nontext_and_adapters(tmp_path):
    generator = TransformersGenerator(model_path=str(tmp_path))
    with pytest.raises(ValueError, match="attachments"):
        generator.generate([], settings={}, attachments=["image.png"])
    with pytest.raises(UnsupportedModelAdapterError):
        generator.generate([], settings={}, model_adapter=object())
    with pytest.raises(ValueError, match="text messages"):
        generator.generate([Message("user", ({"type": "text", "text": "q"},))], settings={})
