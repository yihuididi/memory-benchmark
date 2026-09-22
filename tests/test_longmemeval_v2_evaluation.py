from dataclasses import replace
from types import SimpleNamespace
import json

import pytest

from memory_bench.benchmarks.longmemeval_v2 import harness, metrics
from memory_bench.benchmarks.longmemeval_v2 import LongMemEvalV2Benchmark
from memory_bench.benchmarks.longmemeval_v2 import Judge, validate_eval_spec
from memory_bench.types import EvaluationCase, Prediction, Request


def case(spec, reference, category="static-environment"):
    return EvaluationCase(Request("random-id", "Which option applies?"), reference,
                          {"eval_function": spec, "question_type": category, "domain": "web"}, id="private-id")


@pytest.mark.parametrize("spec,reference,answer,expected", [
    ("norm_phrase_set_match", "red shoes; blue_hat", r"Details. \boxed{Blue-hat and red shoes}", 1),
    ("norm_phrase_set_match", "cat", "concatenate", 0),
    ("norm_phrase_set_match|lower=false", "CAT", "cat", 0),
    ("norm_phrase_set_match|strip_punct=false", "a.b", "a.b", 1),
    ("norm_phrase_set_match_ordered|separators=;", "open;save;open", "open save", 0),
    ("norm_phrase_set_match_ordered|separators=;", "open;save", "save open", 0),
    ("norm_phrase_set_match_ordered|separators=;", "open;save", "first open then save", 1),
    ("mc_choice_match", "B", r"\boxed{A} reconsidered \boxed{Option B.}", 1),
    ("mc_choice_match", "B", "B and C", 0),
    ("mc_choice_set_match", "AC", r"\boxed{Options C and A}", 1),
    ("mc_choice_set_match", "AC", "A", 0),
    ("norm_phrase_set_match", "UNKNOWN", r"\boxed{UNKNOWN}", 0),
    ("norm_phrase_set_match", "a {nested} value", r"\boxed{a {nested} value}", 1),
    ("norm_phrase_set_match", "hello", "", 0),
])
def test_official_deterministic_scoring(tmp_path, spec, reference, answer, expected):
    benchmark = LongMemEvalV2Benchmark(data_root=tmp_path)
    score = benchmark.score(case(spec, reference), Prediction(answer))
    assert score["score"] == expected
    assert benchmark.evaluation_details()["response_parsed_boxed"] == metrics.extract_boxed_answer(answer)


@pytest.mark.parametrize("spec", ["eval_from_spec", "_create_openai_client", "llm_gotchas_checker|evaluator_base_url=http://untrusted", "mc_choice_match|question_item=bad"])
def test_dataset_cannot_select_arbitrary_callables_or_judge_configuration(spec):
    with pytest.raises(ValueError, match="Unsupported"):
        validate_eval_spec(spec)


def fake_client(text, calls, closed):
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           close=lambda: closed.append(True))


@pytest.mark.parametrize("name,category,builder", [
    ("llm_abstention_checker", "static-environment-abs", metrics._build_abstention_judge_messages),
    ("llm_gotchas_checker", "errors-gotchas", metrics._build_gotchas_judge_messages),
])
def test_judge_receives_official_rubric_full_response_and_private_reference(tmp_path, monkeypatch, name, category, builder):
    calls, closed = [], []
    monkeypatch.setenv("TEST_JUDGE_KEY", "key")
    monkeypatch.setattr("memory_bench.generators.openai_compatible.create_client",
                        lambda **kwargs: fake_client('{"label": 1, "reason": "Correct insight"}', calls, closed))
    from memory_bench.generators.openai_compatible import OpenAICompatibleGenerator
    generator = OpenAICompatibleGenerator(model="test-judge", base_url="http://localhost:1234/v1", api_key_env="TEST_JUDGE_KEY")
    benchmark = LongMemEvalV2Benchmark(data_root=tmp_path)
    benchmark.bind_generation(generator, {"reasoning_effort": "medium", "max_completion_tokens": 4096})
    item = case(name, "private reference", category)
    prediction = Prediction(r"Full explanation. \boxed{Final insight}")
    assert benchmark.score(item, prediction)["score"] == 1
    request = calls[0]
    assert request["messages"] == builder(question_text=item.request.question, reference_answer=item.reference,
                                           model_full_response=prediction.answer, model_final_answer="Final insight")
    assert request["model"] == "test-judge"
    assert request["reasoning_effort"] == "medium"
    assert request["max_completion_tokens"] == 4096
    assert benchmark.evaluation_details()["judge"]["reason"] == "Correct insight"
    # Explicit UNKNOWN overrides a positive judge label, exactly as upstream.
    assert benchmark.score(item, Prediction(r"\boxed{UNKNOWN}"))["score"] == 0
    benchmark.close()
    assert closed == []
    generator.close()
    assert closed == [True]


def test_invalid_judgement_raises_instead_of_becoming_a_wrong_answer(monkeypatch):
    from memory_bench.generators.openai_compatible import OpenAICompatibleGenerator
    generator = OpenAICompatibleGenerator(model="test", base_url="http://localhost:1234/v1")
    judge = Judge(generator, {})
    monkeypatch.setattr("memory_bench.generators.openai_compatible.create_client",
                        lambda **kwargs: fake_client("I cannot produce a label", [], []))
    with pytest.raises(ValueError, match="Could not parse"):
        judge.score("llm_gotchas_checker", question="q", reference="r", raw="a", parsed="a")
    judge.close()
    generator.close()


def test_llm_judge_requires_configuration_without_contacting_api(tmp_path):
    benchmark = LongMemEvalV2Benchmark(data_root=tmp_path)
    with pytest.raises(ValueError, match="Configure benchmarks.generation"):
        benchmark._get_judge()


def test_aggregate_matches_upstream_denominators_and_undefined_categories(tmp_path):
    benchmark = LongMemEvalV2Benchmark(data_root=tmp_path)
    scores = [benchmark.score(case("mc_choice_match", "A"), Prediction(answer)) for answer in ("A", "B", "UNKNOWN")]
    # One additional abstention question is correct, making combined static 2/4.
    scores.append({"score": 1., "is_unknown": 0., "is_abstention_problem": 1., "category_index": 1.})
    for index, score in enumerate(scores):
        score["memory_query_duration_seconds"] = float(index + 1)
        score["reader_duration_seconds"] = 100.0
    result = benchmark.aggregate(scores)
    assert result["overall_full_set"] == .5
    assert result["static_accuracy"] == .5
    assert result["overall.overall_non_abstention_only"] == pytest.approx(1 / 3)
    assert result["overall.overall_abstention_only"] == 1
    assert result["combined_abstention_by_category.static.pct_unknown"] == .25
    assert result["non_abstention_by_category.dynamic.count"] == 0
    assert "dynamic_accuracy" not in result
    assert result["memory_query_avg_seconds"] == 2.5
    assert result["memory_query.median_seconds"] == 3  # Upstream uses upper middle.
    assert result["memory_query.p95_seconds"] == 4
    assert result["reader.avg_seconds"] == 100
