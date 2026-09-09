from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from memory_bench.benchmarks.local_judge import BinaryJudgement, LocalJudge
from memory_bench.benchmarks.longmemeval_v2 import make_judge


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
        BinaryJudgement.model_validate_json(raw)


def test_local_judge_never_downloads_missing_model(tmp_path):
    judge = LocalJudge(model_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Local judge model missing"):
        judge.preflight()


def test_model_selection_is_explicit():
    with pytest.raises(ValueError, match="Unsupported evaluator backend"):
        make_judge({"backend": "guess"})
    with pytest.raises(TypeError, match="model_path"):
        make_judge({"backend": "transformers"})


def test_local_judge_wraps_transformers_model_passes_schema_and_reuses_it(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    outlines = pytest.importorskip("outlines")
    (tmp_path / "config.json").write_text('{}')
    calls = []
    # A small actual HF model exercises the Outlines tokenizer/backend conversion.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    raw_tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2}, unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer, unk_token="[UNK]", eos_token="[EOS]",
        chat_template="{% for m in messages %}{{ m.content }}{% endfor %}", model_max_length=4096,
    )
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=3, n_layer=1, n_head=1, n_embd=8, n_positions=4096,
    ))
    def load_model(*args, **kwargs):
        calls.append(("model", kwargs))
        return model
    def load_tokenizer(*args, **kwargs):
        calls.append(("tokenizer", kwargs))
        return tokenizer
    def generator(wrapped, schema):
        assert schema is BinaryJudgement
        def generate(prompt, **kwargs):
            calls.append(("generate", kwargs))
            return '{"label": 1, "reason": "Correct insight"}'
        return generate
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(outlines, "Generator", generator)
    judge = LocalJudge(model_path=str(tmp_path), device="cpu", max_new_tokens=80)
    for _ in range(2):
        assert judge.score("llm_gotchas_checker", question="q", reference="private", raw="a", parsed="a")
    assert sum(name == "model" for name, _ in calls) == 1
    for name, kwargs in calls:
        if name in ("model", "tokenizer"):
            assert kwargs["local_files_only"] is True
            assert kwargs["trust_remote_code"] is False
        else:
            assert kwargs == {"max_new_tokens": 80, "do_sample": False}
    assert judge.details["reason"] == "Correct insight"
    judge.close()
    assert judge.model is judge.tokenizer is judge.generator is None


def test_local_judge_does_not_truncate_or_score_overflow(monkeypatch, tmp_path):
    judge = LocalJudge(model_path=str(tmp_path), device="cpu", max_new_tokens=30)
    monkeypatch.setattr(judge, "_load", lambda: None)
    judge.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100))
    judge.tokenizer = SimpleNamespace(model_max_length=100,
                                     apply_chat_template=lambda *a, **kw: "prompt",
                                     encode=lambda *a, **kw: [1] * 80)
    judge.generator = lambda *a, **kw: pytest.fail("Overflow must fail before inference")
    with pytest.raises(ValueError, match="never truncated"):
        judge.score("llm_gotchas_checker", question="q", reference="r", raw="a", parsed="a")
    assert judge.details == {}


def test_local_judge_invalid_generation_is_not_a_zero_score(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    judge = LocalJudge(model_path=str(tmp_path), device="cpu", max_new_tokens=30)
    monkeypatch.setattr(judge, "_load", lambda: None)
    judge.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100))
    judge.tokenizer = SimpleNamespace(model_max_length=100,
                                     apply_chat_template=lambda *a, **kw: "prompt",
                                     encode=lambda *a, **kw: [1])
    judge.generator = lambda *a, **kw: '{"label": 1'
    with pytest.raises(ValueError, match="invalid/incomplete JSON"):
        judge.score("llm_gotchas_checker", question="q", reference="r", raw="a", parsed="a")
