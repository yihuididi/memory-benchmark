"""Official LongMemEval-V2 scoring and shared evaluation helpers."""
from __future__ import annotations
import base64
import json
import math
import mimetypes
import os
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple


DEFAULT_SEPARATORS: Sequence[str] = (",", ";")
_ABSTENTION_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for flawed-premise (abstention) questions. "
    "Judge whether a model answer correctly identifies that the question premise is wrong, "
    "consistent with the reference answer. "
    "If the model follows the flawed premise and gives a concrete answer under that premise, "
    "it must be graded 0. "
    "If the model's final answer is just UNKNOWN / cannot determine without identifying the flaw, grade 0. "
    "If the model is contradictory (both rejects premise and also gives a concrete premise-following answer), grade 0. "
    "Paraphrases are allowed when they preserve the same core flaw described by the reference answer."
)
_GOTCHAS_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for gotchas-style insight questions. "
    "The reference answer describes the key insight(s). "
    "Grade 1 if the model response includes at least one correct insight point from the reference answer "
    "(paraphrase allowed), and does not contradict any reference point. "
    "If the model's direction is wrong, or it contains contradictions against any reference point, grade 0. "
    "If the model gives multiple points, partial coverage is enough for 1 as long as no contradictions appear."
)
OPENAI_MAX_RETRIES = 10


def normalize_phrase(
    text: str | None,
    *,
    lower: bool = True,
    normalize_hyphen: bool = True,
    strip_punct: bool = True,
) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if lower:
        text = text.lower()
    if normalize_hyphen:
        text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[,;]", " ", text)
    if strip_punct:
        text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_phrases(
    text: str | None,
    *,
    separators: Iterable[str] = DEFAULT_SEPARATORS,
    **normalize_kwargs: bool,
) -> List[str]:
    if text is None:
        return []
    separator_list = list(separators)
    if not separator_list:
        normalized = normalize_phrase(text, **normalize_kwargs)
        return [normalized] if normalized else []
    pattern = "|".join(re.escape(sep) for sep in separator_list)
    parts = re.split(pattern, text)
    normalized_parts = [
        normalize_phrase(part, **normalize_kwargs) for part in parts
    ]
    return [part for part in normalized_parts if part]


def norm_phrase_set_match(
    prediction: str | None,
    answer: str | None,
    *,
    separators: Iterable[str] = DEFAULT_SEPARATORS,
    require_non_empty: bool = True,
    **normalize_kwargs: bool,
) -> bool:
    normalized_pred = normalize_phrase(prediction, **normalize_kwargs)
    answer_phrases = split_phrases(answer, separators=separators, **normalize_kwargs)
    if require_non_empty and (not normalized_pred or not answer_phrases):
        return False
    for phrase in set(answer_phrases):
        pattern = r"\b%s\b" % re.escape(phrase)
        if re.search(pattern, normalized_pred) is None:
            return False
    return True


def norm_phrase_set_match_ordered(
    prediction: str | None,
    answer: str | None,
    *,
    separators: Iterable[str] = DEFAULT_SEPARATORS,
    require_non_empty: bool = True,
    **normalize_kwargs: bool,
) -> bool:
    normalized_pred = normalize_phrase(prediction, **normalize_kwargs)
    answer_phrases = split_phrases(answer, separators=separators, **normalize_kwargs)
    if require_non_empty and (not normalized_pred or not answer_phrases):
        return False
    start = 0
    for phrase in answer_phrases:
        pattern = r"\b%s\b" % re.escape(phrase)
        match = re.search(pattern, normalized_pred[start:])
        if match is None:
            return False
        start += match.end()
    return True


def mc_choice_match(
    prediction: str | None,
    answer: str | None,
    *,
    strip_chars: str = ".",
    require_non_empty: bool = True,
    **_: Any,
) -> bool:
    if prediction is None or answer is None:
        return False
    if not isinstance(prediction, str):
        prediction = str(prediction)
    if not isinstance(answer, str):
        answer = str(answer)
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", prediction.lower())
    candidate = boxed_match.group(1) if boxed_match else prediction
    cleaned = re.sub(r"\b(choice|option)\b", "", candidate, flags=re.IGNORECASE)
    for ch in strip_chars:
        cleaned = cleaned.replace(ch, "")
    cleaned = cleaned.strip().upper()
    expected = answer.strip().upper()
    if require_non_empty and (not cleaned or not expected):
        return False
    return cleaned == expected


_MULTI_SELECT_FILLER_WORDS = {
    "AND",
    "ANSWER",
    "ANSWERS",
    "CHOICE",
    "CHOICES",
    "FINAL",
    "LETTER",
    "LETTERS",
    "OPTION",
    "OPTIONS",
}


def _extract_multi_select_letters(text: str | None) -> list[str]:
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    chunks = re.findall(r"[A-Z]+", text.upper())
    letters: list[str] = []
    for chunk in chunks:
        if chunk in _MULTI_SELECT_FILLER_WORDS:
            continue
        letters.extend(list(chunk))
    return letters


def mc_choice_set_match(
    prediction: str | None,
    answer: str | None,
    *,
    require_non_empty: bool = True,
    **_: Any,
) -> bool:
    pred_letters = _extract_multi_select_letters(prediction)
    answer_letters = _extract_multi_select_letters(answer)
    if require_non_empty and (not pred_letters or not answer_letters):
        return False
    return set(pred_letters) == set(answer_letters)


def extract_boxed_answer(text: str) -> str:
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip()
    i = idx + len(marker)
    depth = 1
    out: List[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    parsed = "".join(out).strip()
    return parsed if parsed else text.strip()


def is_unknown(parsed_answer: str) -> bool:
    return parsed_answer.strip().lower() == "unknown"


def eval_name(eval_spec: str) -> str:
    return eval_spec.split("|", 1)[0].strip()


def score_to_bool(score: Any) -> bool:
    if isinstance(score, bool):
        return score
    if isinstance(score, (int, float)) and score in {0, 1, 0.0, 1.0}:
        return bool(score)
    raise RuntimeError(f"Eval function returned non-binary score: {score!r}")


def llm_abstention_checker(
    prediction: str | None,
    answer: str | None,
    *,
    question_item: dict[str, Any] | None = None,
    parsed_prediction: str | None = None,
    model_response: str | None = None,
    evaluator_model: str | None = None,
    evaluator_base_url: str | None = None,
    evaluator_api_key: str | None = None,
    evaluator_api_key_env: str = "OPENAI_API_KEY",
    evaluator_reasoning_effort: str | None = None,
    evaluator_max_completion_tokens: int = 2048,
    evaluator_temperature: float | None = None,
    evaluator_top_p: float | None = None,
    evaluator_timeout_seconds: float = 43200.0,
    require_non_empty: bool = True,
    **_: Any,
) -> bool:
    prediction_text = _stringify_text(prediction)
    answer_text = _stringify_text(answer)
    if require_non_empty and (not prediction_text or not answer_text):
        return False

    if not evaluator_model:
        raise ValueError(
            "llm_abstention_checker requires evaluator_model. "
            "Pass evaluator_model via eval_from_spec overrides."
        )

    if evaluator_api_key is None:
        evaluator_api_key = os.getenv(evaluator_api_key_env)
    if evaluator_base_url and not evaluator_api_key:
        evaluator_api_key = "EMPTY"
    if not evaluator_base_url and not evaluator_api_key:
        raise ValueError(
            "llm_abstention_checker requires evaluator_api_key (or set evaluator_api_key_env)."
        )

    question_text = _extract_question_text(question_item)
    final_answer_text = _stringify_text(parsed_prediction) or prediction_text
    full_response_text = _stringify_text(model_response) or prediction_text
    if require_non_empty and not final_answer_text:
        return False

    client = _create_openai_client(
        base_url=evaluator_base_url,
        api_key=evaluator_api_key,
    )
    messages = _build_abstention_judge_messages(
        question_text=question_text,
        reference_answer=answer_text,
        model_full_response=full_response_text,
        model_final_answer=final_answer_text,
    )
    judge_text = _call_chat_completion(
        client=client,
        model=evaluator_model,
        messages=messages,
        max_completion_tokens=evaluator_max_completion_tokens,
        reasoning_effort=evaluator_reasoning_effort,
        temperature=evaluator_temperature,
        top_p=evaluator_top_p,
        timeout_seconds=evaluator_timeout_seconds,
    )
    label, _reason = _parse_llm_binary_judgement(judge_text)
    return label == 1


def llm_gotchas_checker(
    prediction: str | None,
    answer: str | None,
    *,
    question_item: dict[str, Any] | None = None,
    parsed_prediction: str | None = None,
    model_response: str | None = None,
    evaluator_model: str | None = None,
    evaluator_base_url: str | None = None,
    evaluator_api_key: str | None = None,
    evaluator_api_key_env: str = "OPENAI_API_KEY",
    evaluator_reasoning_effort: str | None = None,
    evaluator_max_completion_tokens: int = 2048,
    evaluator_temperature: float | None = None,
    evaluator_top_p: float | None = None,
    evaluator_timeout_seconds: float = 43200.0,
    require_non_empty: bool = True,
    **_: Any,
) -> bool:
    prediction_text = _stringify_text(prediction)
    answer_text = _stringify_text(answer)
    if require_non_empty and (not prediction_text or not answer_text):
        return False

    if not evaluator_model:
        raise ValueError(
            "llm_gotchas_checker requires evaluator_model. "
            "Pass evaluator_model via eval_from_spec overrides."
        )

    if evaluator_api_key is None:
        evaluator_api_key = os.getenv(evaluator_api_key_env)
    if evaluator_base_url and not evaluator_api_key:
        evaluator_api_key = "EMPTY"
    if not evaluator_base_url and not evaluator_api_key:
        raise ValueError(
            "llm_gotchas_checker requires evaluator_api_key (or set evaluator_api_key_env)."
        )

    question_text = _extract_question_text(question_item)
    final_answer_text = _stringify_text(parsed_prediction) or prediction_text
    full_response_text = _stringify_text(model_response) or prediction_text
    if require_non_empty and not final_answer_text:
        return False

    client = _create_openai_client(
        base_url=evaluator_base_url,
        api_key=evaluator_api_key,
    )
    messages = _build_gotchas_judge_messages(
        question_text=question_text,
        reference_answer=answer_text,
        model_full_response=full_response_text,
        model_final_answer=final_answer_text,
    )
    judge_text = _call_chat_completion(
        client=client,
        model=evaluator_model,
        messages=messages,
        max_completion_tokens=evaluator_max_completion_tokens,
        reasoning_effort=evaluator_reasoning_effort,
        temperature=evaluator_temperature,
        top_p=evaluator_top_p,
        timeout_seconds=evaluator_timeout_seconds,
    )
    label, _reason = _parse_llm_binary_judgement(judge_text)
    return label == 1


def _stringify_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_question_text(question_item: dict[str, Any] | None) -> str:
    if not isinstance(question_item, dict):
        return ""
    question = question_item.get("question")
    if isinstance(question, str):
        return question.strip()
    if isinstance(question, dict):
        text = question.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def _build_abstention_judge_messages(
    *,
    question_text: str,
    reference_answer: str,
    model_full_response: str,
    model_final_answer: str,
) -> list[dict[str, str]]:
    user_prompt = (
        "Evaluate whether the model correctly handled a flawed-premise question.\n\n"
        "Question:\n"
        f"{question_text}\n\n"
        "Reference answer (the expected flaw-aware conclusion):\n"
        f"{reference_answer}\n\n"
        "Model full response:\n"
        f"{model_full_response}\n\n"
        "Model extracted final answer:\n"
        f"{model_final_answer}\n\n"
        "Scoring rubric:\n"
        "- Label 1 only if the model clearly identifies the flawed premise and reaches a conclusion "
        "equivalent to the reference answer.\n"
        "- Label 1 also if the model explicitly says it lacks access to the user's specific "
        "live environment/instance/configuration and therefore cannot verify, provided it does not "
        "give a concrete premise-following answer.\n"
        "- Label 0 if the model follows the flawed premise and gives a concrete answer under that premise.\n"
        "- Label 0 for generic UNKNOWN/insufficient-info replies that do not identify a flaw and do not "
        "make the explicit environment-access limitation clear.\n"
        "- Label 0 if contradictory.\n\n"
        "Output JSON only:\n"
        '{"label": 0 or 1, "reason": "short rationale"}'
    )
    return [
        {"role": "system", "content": _ABSTENTION_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_gotchas_judge_messages(
    *,
    question_text: str,
    reference_answer: str,
    model_full_response: str,
    model_final_answer: str,
) -> list[dict[str, str]]:
    user_prompt = (
        "Evaluate whether the model answer captures the gotcha insight.\n\n"
        "Question:\n"
        f"{question_text}\n\n"
        "Reference answer (insight points):\n"
        f"{reference_answer}\n\n"
        "Model full response:\n"
        f"{model_full_response}\n\n"
        "Model extracted final answer:\n"
        f"{model_final_answer}\n\n"
        "Scoring rubric:\n"
        "- Label 1 if the model includes at least one correct insight point from the reference answer "
        "(paraphrase acceptable), and does not contradict any reference point.\n"
        "- Label 1 even if only part of a multi-point reference answer is covered, as long as there is "
        "no contradiction.\n"
        "- Label 0 if direction is wrong (suggests opposite action/cause), even if some wording overlaps.\n"
        "- Label 0 if any point in the model response contradicts any reference point.\n"
        "- Label 0 if the response is irrelevant or generic without insight.\n\n"
        "Output JSON only:\n"
        '{"label": 0 or 1, "reason": "short rationale"}'
    )
    return [
        {"role": "system", "content": _GOTCHAS_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _parse_llm_binary_judgement(text: str) -> Tuple[int, str]:
    cleaned = _strip_markdown_code_fence(_stringify_text(text))
    if not cleaned:
        raise ValueError("Empty judgement response from evaluator model.")

    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        json_blob = json_match.group(0)
        try:
            payload = json.loads(json_blob)
            if not isinstance(payload, dict):
                raise ValueError("Evaluator JSON payload must be an object.")
            label = payload.get("label")
            if label in {0, 1, "0", "1"}:
                label_int = int(label)
                reason = _stringify_text(payload.get("reason"))
                return label_int, reason
        except json.JSONDecodeError:
            # Fall through to regex-based extraction for non-strict JSON-like outputs.
            pass

    label_match = re.search(r'"label"\s*:\s*([01])', cleaned, flags=re.IGNORECASE)
    if not label_match:
        label_match = re.search(r"'label'\s*:\s*([01])", cleaned, flags=re.IGNORECASE)
    if not label_match:
        label_match = re.search(r"\blabel\b\s*[:=]\s*([01])", cleaned, flags=re.IGNORECASE)
    if label_match:
        return int(label_match.group(1)), cleaned

    raise ValueError(f"Could not parse evaluator binary judgement: {cleaned!r}")


def _create_openai_client(*, base_url: str | None, api_key: str | None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required for llm_abstention_checker."
        ) from exc

    if base_url:
        if not api_key:
            api_key = "EMPTY"
        return OpenAI(base_url=base_url, api_key=api_key, max_retries=OPENAI_MAX_RETRIES)
    return OpenAI(api_key=api_key, max_retries=OPENAI_MAX_RETRIES)


def _call_chat_completion(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    reasoning_effort: str | None,
    temperature: float | None,
    top_p: float | None,
    timeout_seconds: float,
) -> str:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "timeout": timeout_seconds,
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        request["temperature"] = temperature
    if top_p is not None:
        request["top_p"] = top_p

    response = client.chat.completions.create(**request)
    message_content = response.choices[0].message.content
    if isinstance(message_content, str):
        return message_content.strip()
    if isinstance(message_content, list):
        text_parts = []
        for item in message_content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        joined = "\n".join(text_parts).strip()
        if joined:
            return joined
    raise ValueError("Evaluator model returned empty response content.")


def _parse_eval_value(key: str, value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if key in {"separators", "separator"}:
        if not value:
            return []
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return json.loads(stripped)
        return [ch for ch in value if not ch.isspace()]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_eval_function_spec(spec: str) -> tuple[Callable[..., Any], dict[str, Any]]:
    if not spec or not isinstance(spec, str):
        raise ValueError("eval function spec must be a non-empty string.")
    parts = [part.strip() for part in spec.split("|")]
    name = parts[0]
    if not name:
        raise ValueError("eval function spec missing function name.")

    func = globals().get(name)
    if func is None or not callable(func):
        raise ValueError(f"Unknown eval function: {name}")

    kwargs: dict[str, Any] = {}
    for part in parts[1:]:
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid eval function option: {part}")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid eval function option: {part}")
        if key in kwargs:
            raise ValueError(f"Duplicate eval function option: {key}")
        kwargs[key] = _parse_eval_value(key, value)

    return func, kwargs


def eval_from_spec(spec: str, *args: Any, **overrides: Any) -> Any:
    func, kwargs = parse_eval_function_spec(spec)
    kwargs.update(overrides)
    return func(*args, **kwargs)
"""Selected upstream harness helpers; see NOTICE.md for provenance and changes."""
import base64
import mimetypes
import threading
from pathlib import Path
from typing import Any, Tuple

MemoryContextItem = dict[str, str]
MEMORY_CONTEXT_PROCESSOR_LOCAL = threading.local()

CATEGORY_MAP = {
    "static-environment": "static",
    "static-environment-abs": "static-abs",
    "dynamic-environment": "dynamic",
    "dynamic-environment-abs": "dynamic-abs",
    "procedure": "procedure",
    "procedure-abs": "procedure-abs",
    "errors-gotchas": "gotchas",
}


NON_ABSTENTION_CATEGORIES = ["static", "dynamic", "procedure", "gotchas"]


ABSTENTION_CATEGORIES = ["static-abs", "dynamic-abs", "procedure-abs"]


COMBINED_ABSTENTION_CATEGORY_PAIRS = {
    "static": ("static", "static-abs"),
    "dynamic": ("dynamic", "dynamic-abs"),
    "procedure": ("procedure", "procedure-abs"),
}


LLM_EVAL_FUNCTIONS = {"llm_abstention_checker", "llm_gotchas_checker"}


DOMAIN_SYSTEM_PROMPTS = {
    "web": (
        "You are an experienced colleague in a web browsing environment that has "
        "a customized magento-based shopping website, a customized magento-based "
        "shopping admin cms website, as well as a customized forum website based "
        "on reddit/postmill. Answer based on your memory of the environment. "
        "If you do not know the answer, output exactly \\boxed{UNKNOWN}. "
        "Do not guess. Never attempt to guess an answer if you are not sure. "
        "If you believe the question's construction/premise is wrong, provide an "
        "explanation in \boxed{} explaining why the question is flawed."
    ),
    "enterprise": (
        "You are an experienced colleague working in a customized ServiceNow "
        "environment. Answer based on your memory of the environment. "
        "If you do not know the answer, output exactly \\boxed{UNKNOWN}. "
        "Do not guess. Never attempt to guess an answer if you are not sure. "
        "If you believe the question's construction/premise is wrong, provide an "
        "explanation in \boxed{} explaining why the question is flawed."
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_memory_context_items(
    memory_context: Any,
    *,
    question_id: str,
) -> list[MemoryContextItem]:
    require(isinstance(memory_context, list), f"memory.query must return a list for {question_id}")
    validated: list[MemoryContextItem] = []
    for idx, item in enumerate(memory_context):
        require(
            isinstance(item, dict),
            f"memory.query item {idx} must be an object for {question_id}",
        )
        item_type = item.get("type")
        value = item.get("value")
        require(
            item_type in {"text", "image"},
            f"memory.query item {idx} has invalid type {item_type!r} for {question_id}",
        )
        require(
            isinstance(value, str) and value.strip(),
            f"memory.query item {idx} has invalid value for {question_id}",
        )
        if item_type == "image":
            require(Path(value).exists(), f"memory.query image path does not exist for {question_id}: {value}")
        validated.append({"type": item_type, "value": value})
    return validated


def get_memory_context_processor() -> Any:
    from transformers import AutoProcessor

    processor = getattr(MEMORY_CONTEXT_PROCESSOR_LOCAL, "processor", None)
    if processor is None:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
        MEMORY_CONTEXT_PROCESSOR_LOCAL.processor = processor
    return processor


def load_memory_context_images(memory_context: list[MemoryContextItem]) -> list[Any | None]:
    from PIL import Image

    loaded_images: list[Any | None] = []
    for item in memory_context:
        if item["type"] != "image":
            loaded_images.append(None)
            continue
        with Image.open(item["value"]) as image:
            loaded_images.append(image.convert("RGB"))
    return loaded_images


def count_memory_context_tokens(
    memory_context: list[MemoryContextItem],
    loaded_images: list[Any | None],
    *,
    processor: Any = None,
) -> int:
    require(
        len(memory_context) == len(loaded_images),
        "memory_context and loaded_images must have the same length",
    )
    if not memory_context:
        return 0

    if processor is None:
        processor = get_memory_context_processor()
    content_parts: list[dict[str, str]] = []
    images: list[Any] = []
    for item, loaded_image in zip(memory_context, loaded_images):
        if item["type"] == "text":
            content_parts.append({"type": "text", "text": item["value"]})
            continue
        require(loaded_image is not None, "Missing loaded image for memory context item")
        content_parts.append({"type": "image"})
        images.append(loaded_image)

    prompt_text = processor.apply_chat_template(
        [{"role": "user", "content": content_parts}],
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = processor(
        text=prompt_text,
        images=images or None,
        return_tensors="pt",
    )
    return int(encoded["input_ids"].shape[-1])


def truncate_memory_context(
    memory_context: list[MemoryContextItem],
    *,
    max_tokens: int,
    question_id: str,
    processor: Any = None,
) -> tuple[list[MemoryContextItem], int, int]:
    require(max_tokens > 0, "memory_context_max_tokens must be positive")
    loaded_images = load_memory_context_images(memory_context)
    prefix_token_counts: dict[int, int] = {0: 0}

    def prefix_token_count(prefix_length: int) -> int:
        require(
            0 <= prefix_length <= len(memory_context),
            f"Invalid memory context prefix length {prefix_length} for {question_id}",
        )
        if prefix_length not in prefix_token_counts:
            prefix_token_counts[prefix_length] = count_memory_context_tokens(
                memory_context[:prefix_length],
                loaded_images[:prefix_length],
                processor=processor,
            )
        return prefix_token_counts[prefix_length]

    original_token_count = prefix_token_count(len(memory_context))
    if original_token_count <= max_tokens:
        return memory_context, original_token_count, original_token_count

    lo = 0
    hi = len(memory_context)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if prefix_token_count(mid) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1

    truncated_items = memory_context[:lo]
    truncated_token_count = prefix_token_count(lo)
    print(
        f"Truncated memory context for {question_id}: "
        f"original_tokens={original_token_count} "
        f"truncated_tokens={truncated_token_count} "
        f"original_items={len(memory_context)} "
        f"truncated_items={len(truncated_items)}"
    )
    return truncated_items, original_token_count, truncated_token_count


def to_data_url(image_path: str) -> str:
    path = Path(image_path)
    require(path.exists(), f"Missing image file: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime or 'image/png'};base64,{encoded}"


def build_messages(
    system_prompt: str,
    question_text: str,
    image_path: str | None,
    memory_context: list[MemoryContextItem],
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    intro_text = "### Memory context:\n"
    if not memory_context:
        intro_text += "(empty)"
    content_parts: list[dict[str, Any]] = [{"type": "text", "text": intro_text}]
    log_parts: list[dict[str, Any]] = [{"type": "text", "text": intro_text}]
    for item in memory_context:
        if item["type"] == "text":
            content_parts.append({"type": "text", "text": item["value"]})
            log_parts.append({"type": "text", "text": item["value"]})
        else:
            content_parts.append({"type": "image_url", "image_url": {"url": to_data_url(item["value"])}})
            log_parts.append({"type": "image_path", "image_path": item["value"]})
    question_block = f"\n\n### Question to answer:\n{question_text}"
    content_parts.append({"type": "text", "text": question_block})
    log_parts.append({"type": "text", "text": question_block})
    if image_path is not None:
        content_parts.append({"type": "image_url", "image_url": {"url": to_data_url(image_path)}})
        log_parts.append({"type": "image_path", "image_path": image_path})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts},
    ]
    messages_for_log = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": log_parts},
    ]
    return messages, messages_for_log


def extract_text_from_response_message(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_val = item.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val.strip())
            else:
                text_val = getattr(item, "text", None)
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val.strip())
        joined = "\n".join(parts).strip()
        if joined:
            return joined
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


def extract_usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def category_from_question_type(question_type: str) -> str:
    require(question_type in CATEGORY_MAP, f"Unexpected question_type: {question_type}")
    return CATEGORY_MAP[question_type]


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    require(records, "No records to aggregate")

    non_abst = [r for r in records if not r["is_abstention_problem"]]
    abst = [r for r in records if r["is_abstention_problem"]]

    def mean_score(rows: list[dict[str, Any]]) -> float | None:
        if not rows:
            return None
        return sum(float(r["score"]) for r in rows) / len(rows)

    def breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {
                "count": 0,
                "pct_correct": None,
                "pct_answered_wrong": None,
                "pct_unknown": None,
            }
        unknown_count = sum(1 for r in rows if r["is_unknown"])
        correct_count = sum(1 for r in rows if r["score_bool"] and not r["is_unknown"])
        wrong_count = n - correct_count - unknown_count
        return {
            "count": n,
            "pct_correct": correct_count / n,
            "pct_answered_wrong": wrong_count / n,
            "pct_unknown": unknown_count / n,
        }

    overall = {
        "overall_full_set": mean_score(records),
        "overall_non_abstention_only": mean_score(non_abst),
        "overall_abstention_only": mean_score(abst),
        "count_all_questions": len(records),
        "count_non_abstention": len(non_abst),
        "count_abstention": len(abst),
    }

    non_abstention_by_category: dict[str, Any] = {}
    for cat in NON_ABSTENTION_CATEGORIES:
        rows = [r for r in non_abst if r["category"] == cat]
        non_abstention_by_category[cat] = breakdown(rows)

    abstention_by_category: dict[str, Any] = {}
    for cat in ABSTENTION_CATEGORIES:
        rows = [r for r in abst if r["category"] == cat]
        abstention_by_category[cat] = breakdown(rows)

    combined_abstention_by_category: dict[str, Any] = {}
    for cat, pair in COMBINED_ABSTENTION_CATEGORY_PAIRS.items():
        rows = [r for r in records if r["category"] in pair]
        combined_abstention_by_category[cat] = breakdown(rows)

    return {
        "overall": overall,
        "non_abstention_by_category": non_abstention_by_category,
        "abstention_by_category": abstention_by_category,
        "combined_abstention_by_category": combined_abstention_by_category,
        "abstention_overall": breakdown(abst),
    }


def score_prediction(
    row: dict[str, Any],
    eval_config: dict[str, Any],
) -> tuple[bool, str, bool]:
    q_eval_name = row["eval_name"]
    eval_kwargs: dict[str, Any] = {}
    if q_eval_name in LLM_EVAL_FUNCTIONS:
        eval_kwargs.update(eval_config)
        eval_kwargs["question_item"] = row["question_item"]
        eval_kwargs["parsed_prediction"] = row["response_parsed_boxed"]
        eval_kwargs["model_response"] = row["response_raw"]

    prediction_for_eval = row["response_parsed_boxed"]
    if q_eval_name in LLM_EVAL_FUNCTIONS:
        prediction_for_eval = row["response_raw"]

    score_raw = eval_from_spec(
        row["eval_function"],
        prediction_for_eval,
        row["answer_gold"],
        **eval_kwargs,
    )
    score_bool = score_to_bool(score_raw)
    if row["is_unknown"]:
        score_bool = False
    return score_bool, q_eval_name, row["is_unknown"]

metrics = harness = sys.modules[__name__]

"""Official LME-V2 scoring with explicit, evaluator-only judge configuration.

No dataset field can select an arbitrary callable or override model settings.
Model dependencies are optional; deterministic scoring remains offline.
"""




UPSTREAM_REVISION = "2cc8c540bdb87fe6761629b585e727e1c4704520"
CATEGORIES = tuple(harness.CATEGORY_MAP.values())
_NORMALIZATION = {"lower", "normalize_hyphen", "strip_punct", "require_non_empty"}
_OPTIONS = {
    "norm_phrase_set_match": _NORMALIZATION | {"separators"},
    "norm_phrase_set_match_ordered": _NORMALIZATION | {"separators"},
    "mc_choice_match": {"strip_chars", "require_non_empty"},
    "mc_choice_set_match": {"require_non_empty"},
    "llm_abstention_checker": {"require_non_empty"},
    "llm_gotchas_checker": {"require_non_empty"},
}


def validate_eval_spec(spec: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(spec, str) or metrics.eval_name(spec) not in _OPTIONS:
        raise ValueError(f"Unsupported LongMemEval-V2 evaluator: {spec!r}")
    name = metrics.eval_name(spec)
    _, options = metrics.parse_eval_function_spec(spec)
    for key, value in options.items():
        if key not in _OPTIONS[name]:
            raise ValueError(f"Unsupported option {key!r} for {name}")
        if key in _NORMALIZATION and type(value) is not bool:
            raise ValueError(f"{name}: {key} must be a boolean")
        if key == "separators" and not (
            isinstance(value, list) and all(isinstance(x, str) and x for x in value)
        ):
            raise ValueError(f"{name}: separators must be a list of nonempty strings")
        if key == "strip_chars" and not isinstance(value, str):
            raise ValueError(f"{name}: strip_chars must be a string")
    return name, options


def judge_messages(name: str, *, question: str, reference: str, raw: str, parsed: str):
    builders = {
        "llm_abstention_checker": metrics._build_abstention_judge_messages,
        "llm_gotchas_checker": metrics._build_gotchas_judge_messages,
    }
    return builders[name](question_text=question.strip(), reference_answer=reference.strip(),
                          model_full_response=raw.strip(), model_final_answer=parsed.strip())


@cache
def binary_judgement_schema():
    """Load the strict judgment schema only when LLM evaluation needs it."""
    from pydantic import BaseModel, ConfigDict, Field

    class BinaryJudgement(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        label: int = Field(ge=0, le=1)
        reason: str = Field(min_length=1)

    return BinaryJudgement


class Judge:
    """Benchmark rubric and strict parsing around a borrowed generator."""

    def __init__(self, generator, settings):
        self.generator = generator
        self.settings = copy.deepcopy(dict(settings))
        self.details: dict[str, Any] = {}

    def preflight(self):
        self.generator.preflight()

    def score(self, name: str, **inputs: str) -> bool:
        from memory_bench.types import Message
        schema = binary_judgement_schema()
        self.details = {}
        prediction = self.generator.generate_structured(
            tuple(Message(**m) for m in judge_messages(name, **inputs)),
            settings=copy.deepcopy(self.settings), schema=schema)
        try:
            result = schema.model_validate_json(prediction.answer)
        except ValueError as exc:
            raise ValueError(f"Could not parse evaluator binary judgement: {prediction.answer!r}") from exc
        self.details = {**prediction.metadata, "settings": copy.deepcopy(self.settings),
                        **result.model_dump(), "raw": prediction.answer}
        return result.label == 1

    def close(self):
        # The runner owns the borrowed generator.
        pass


def aggregate_scores(scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
    rows = []
    for score in scores:
        category_index = score["category_index"]
        if category_index != int(category_index) or not 0 <= category_index < len(CATEGORIES):
            raise ValueError(f"Invalid category_index: {category_index}")
        rows.append({**score, "category": CATEGORIES[int(category_index)],
                     "score_bool": bool(score["score"])})
    nested = harness.aggregate_metrics(rows)
    flat: dict[str, float] = {}

    def flatten(obj: dict, prefix: str = "") -> None:
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flatten(value, path)
            elif value is not None:
                flat[path] = float(value)

    flatten(nested)
    flat["overall_full_set"] = flat["overall.overall_full_set"]
    for category in ("static", "dynamic", "procedure"):
        key = f"combined_abstention_by_category.{category}.pct_correct"
        if key in flat:
            flat[f"{category}_accuracy"] = flat[key]
    for category, key in (("gotchas", "non_abstention_by_category.gotchas.pct_correct"),
                          ("abstention", "abstention_overall.pct_correct")):
        if key in flat:
            flat[f"{category}_accuracy"] = flat[key]
    for prefix, field in (("memory_query", "memory_query_duration_seconds"),
                          ("reader", "reader_duration_seconds")):
        durations = sorted(score[field] for score in scores if field in score)
        if durations:
            if len(durations) != len(scores):
                raise ValueError(f"Incomplete {field} measurements")
            if any(not math.isfinite(x) or x < 0 for x in durations):
                raise ValueError(f"Invalid {field} measurements")
            n = len(durations)
            flat.update({f"{prefix}.avg_seconds": sum(durations) / n,
                         f"{prefix}.median_seconds": durations[n // 2],
                         f"{prefix}.p95_seconds": durations[min(n - 1, int(.95 * n))],
                         f"{prefix}.max_seconds": durations[-1],
                         f"{prefix}.total_seconds": sum(durations)})
    if "memory_query.avg_seconds" in flat:
        flat["memory_query_avg_seconds"] = flat["memory_query.avg_seconds"]
    return flat

import copy
import hashlib
import uuid

from memory_bench.benchmarks import register_benchmark
from memory_bench.benchmarks.base import BaseBenchmark
from memory_bench.config import PluginSpec
from memory_bench.types import Episode, EvaluationCase, MemoryRecord, Prediction, Request

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json(text: str, location: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_object)
    except ValueError as exc:
        raise ValueError(f"Invalid JSON in {location}: {exc}") from exc


def _jsonl(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            row = _json(line, location)
            _require(isinstance(row, dict), f"{location}: expected a JSON object")
            yield location, row


def _text(row: Mapping[str, Any], key: str, location: str, *, nonempty: bool = True) -> str:
    value = row.get(key)
    _require(
        isinstance(value, str) and (not nonempty or bool(value.strip())),
        f"{location}: {key} must be {'a nonempty string' if nonempty else 'a string'}",
    )
    return value


@register_benchmark("longmemeval_v2")
class LongMemEvalV2Benchmark(BaseBenchmark):
    supports_scoring = True

    def __init__(
        self,
        *,
        data_root: Path | str,
        tier: str = "small",
        domain: str | None = None,
        limit: int | None = None,
    ) -> None:
        _require(tier in ("small", "medium"), "tier must be 'small' or 'medium'")
        _require(domain in (None, "web", "enterprise"), "domain must be 'web' or 'enterprise'")
        _require(
            limit is None or (type(limit) is int and limit > 0),
            "limit must be a positive integer",
        )
        _require(isinstance(data_root, (str, Path)) and bool(str(data_root).strip()),
                 "data_root must be a nonempty path")
        self.data_root = Path(data_root).expanduser().resolve()
        self.tier = tier
        self.domain = domain
        self.limit = limit
        self._judge = None
        self._details: dict[str, Any] = {}

    def _get_judge(self):
        if self._judge is None:
            if self.generation is None:
                raise ValueError(
                    "LongMemEval-V2 includes LLM-scored questions. Configure benchmarks.generation "
                    "with a named generator. No heuristic fallback is used."
                )
            self._judge = Judge(self.generation, self.generation_settings)
        return self._judge

    def _selected_questions(self) -> list[dict[str, Any]]:
        rows = [row for row in self._questions().values()
                if self.domain is None or row["domain"] == self.domain]
        return rows[:self.limit] if self.limit is not None else rows

    def validate_run(self, agent: PluginSpec, generation: PluginSpec) -> None:
        if agent.type == "longmemeval_v2":
            _require(self.domain is not None and agent.options.get("domain") == self.domain,
                     "The longmemeval_v2 agent domain must match the benchmark domain; "
                     "configure one domain per run, or use your own agent")
        _require(generation.type != "deterministic", "The deterministic demo generator cannot answer "
                 "LongMemEval-V2 questions; configure your agent's model generator")
        rows = self._selected_questions()
        _require(bool(rows), "No questions selected; check the dataset and domain filter")
        needs_judge = False
        for row in rows:
            name, _ = validate_eval_spec(row["eval_function"])
            harness.category_from_question_type(row["question_type"])
            if name == "llm_abstention_checker":
                _require("-abs" in row["question_type"],
                         "llm_abstention_checker requires an -abs question_type")
            needs_judge |= name in harness.LLM_EVAL_FUNCTIONS
        if needs_judge:
            self._get_judge().preflight()

    def _image(self, value: Any, location: str) -> str | None:
        if value is None:
            return None
        _require(isinstance(value, str) and bool(value.strip()),
                 f"{location}: image path must be a nonempty string or null")
        path = (self.data_root / value).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{location}: missing image {path}. Download the full dataset and run "
                "the upstream data/prepare_data.py --mode symlink first."
            )
        return str(path)

    def _questions(self) -> dict[str, dict[str, Any]]:
        questions: dict[str, dict[str, Any]] = {}
        for location, row in _jsonl(self.data_root / "questions.jsonl"):
            question_id = _text(row, "id", location)
            _require(question_id not in questions, f"{location}: duplicate question id {question_id!r}")
            _require(row.get("domain") in ("web", "enterprise"),
                     f"{location}: invalid question domain for {question_id!r}")
            for key in ("environment", "question_type", "question", "eval_function"):
                _text(row, key, location)
            _text(row, "answer", location, nonempty=False)
            questions[question_id] = row
        return questions

    def _haystacks(self, questions: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
        path = self.data_root / "haystacks" / f"lme_v2_{self.tier}.json"
        payload = _json(path.read_text(encoding="utf-8"), str(path))
        _require(isinstance(payload, dict), f"{path}: expected a JSON object of haystacks")
        haystacks: dict[str, tuple[str, ...]] = {}
        for question_id, trajectory_ids in payload.items():
            location = f"{path}: question {question_id!r}"
            _require(question_id in questions, f"{location}: unknown question id")
            _require(isinstance(trajectory_ids, list) and bool(trajectory_ids),
                     f"{location}: haystack must be a nonempty list")
            _require(all(isinstance(item, str) and item.strip() for item in trajectory_ids),
                     f"{location}: haystack trajectory ids must be nonempty strings")
            _require(len(set(trajectory_ids)) == len(trajectory_ids),
                     f"{location}: duplicate trajectory id in haystack")
            haystacks[question_id] = tuple(trajectory_ids)
        missing = questions.keys() - haystacks.keys()
        _require(not missing, f"{path}: missing haystack entries for questions {sorted(missing)[:5]}")
        return haystacks

    def _record(self, row: dict[str, Any], location: str) -> MemoryRecord:
        trajectory_id = row["id"]
        location = f"{location}: trajectory {trajectory_id!r}"
        for key in ("environment", "goal", "start_url"):
            _text(row, key, location, nonempty=False)
        _require(row.get("outcome") in ("success", "failure"),
                 f"{location}: outcome must be 'success' or 'failure'")
        states = row.get("states")
        _require(isinstance(states, list) and bool(states),
                 f"{location}: states must be a nonempty list")
        parts = [f"Goal: {row['goal']}", f"Outcome: {row['outcome']}", f"Start URL: {row['start_url']}"]
        resolved_states: list[dict[str, Any]] = []
        for index, state in enumerate(states):
            state_location = f"{location}, states[{index}]"
            _require(isinstance(state, dict), f"{state_location}: expected a state object")
            state_index = state.get("state_index")
            _require(type(state_index) is int and state_index >= 0,
                     f"{state_location}: state_index must be a nonnegative integer")
            step = state.get("step")
            _require(step is None or (type(step) is int and step >= 0),
                     f"{state_location}: step must be a nonnegative integer or null")
            url = _text(state, "url", state_location, nonempty=False)
            tree = _text(state, "accessibility_tree", state_location, nonempty=False)
            for key in ("action", "thought"):
                _require(state.get(key) is None or isinstance(state[key], str),
                         f"{state_location}: {key} must be a string or null")
            screenshot = self._image(state.get("screenshot"), f"{state_location} screenshot")
            resolved_states.append({**state, "screenshot": screenshot})
            parts.extend([
                "", f"State index: {state_index}", f"Step: {step}", f"URL: {url}",
                f"Action: {state.get('action') or ''}", f"Thought: {state.get('thought') or ''}",
                f"Accessibility tree:\n{tree}",
            ])
        return MemoryRecord(
            id=trajectory_id,
            text="\n".join(parts),
            metadata={**row, "states": resolved_states},
        )

    def load(self) -> Iterable[Episode]:
        """Load selected questions in file order; limit applies after domain filtering.

        JSONL is streamed, keeping only trajectories selected by those questions.
        Images are validated for the selected questions and trajectories. Repeated
        histories share records; memory backends must treat them as read-only.
        """
        questions = self._questions()
        haystacks = self._haystacks(questions)
        selected = [row for row in questions.values() if self.domain is None or row["domain"] == self.domain]
        if self.limit is not None:
            selected = selected[:self.limit]
        _require(bool(selected), "No questions selected; check the dataset and domain filter")

        groups: dict[tuple[str, ...], list[EvaluationCase]] = {}
        required: dict[str, str] = {}
        for row in selected:
            history = haystacks[row["id"]]
            for trajectory_id in history:
                _require(trajectory_id not in required or required[trajectory_id] == row["domain"],
                         f"Cross-domain trajectory {trajectory_id!r} in question {row['id']!r}")
                required[trajectory_id] = row["domain"]
            image = self._image(row.get("image"), f"Question {row['id']!r} image")
            case = EvaluationCase(
                id=row["id"],
                request=Request(
                    id=uuid.uuid4().hex,
                    question=row["question"],
                    attachments=(image,) if image else (),
                ),
                reference=row["answer"],
                metadata={key: value for key, value in row.items() if key not in ("question", "image", "answer")},
            )
            groups.setdefault(history, []).append(case)

        records: dict[str, MemoryRecord] = {}
        seen: set[str] = set()
        path = self.data_root / "trajectories.jsonl"
        for location, row in _jsonl(path):
            trajectory_id = _text(row, "id", location)
            _require(trajectory_id not in seen, f"{location}: duplicate trajectory id {trajectory_id!r}")
            seen.add(trajectory_id)
            if trajectory_id not in required:
                continue
            _require(row.get("domain") == required[trajectory_id],
                     f"{location}: domain mismatch for trajectory {trajectory_id!r}; expected {required[trajectory_id]!r}")
            records[trajectory_id] = self._record(row, location)
        missing = required.keys() - records.keys()
        _require(not missing, f"{path}: unknown trajectory ids referenced by haystacks: {sorted(missing)[:5]}")

        for history, cases in groups.items():
            digest = hashlib.sha256(json.dumps(history, ensure_ascii=False).encode("utf-8")).hexdigest()
            yield Episode(
                id=f"longmemeval-v2-{digest}",
                records=tuple(records[trajectory_id] for trajectory_id in history),
                cases=tuple(cases),
            )

    def score(self, case: EvaluationCase, prediction: Prediction) -> dict[str, float]:
        self._details = {}
        name, options = validate_eval_spec(case.metadata["eval_function"])
        category = harness.category_from_question_type(case.metadata["question_type"])
        _require(isinstance(prediction.answer, str) and isinstance(case.reference, str),
                 "LongMemEval-V2 predictions and references must be strings")
        if name == "llm_abstention_checker":
            _require("-abs" in case.metadata["question_type"],
                     "llm_abstention_checker requires an -abs question_type")
        parsed = metrics.extract_boxed_answer(prediction.answer)
        unknown = metrics.is_unknown(parsed)
        self._details = {"upstream_revision": UPSTREAM_REVISION,
                         "eval_function": case.metadata["eval_function"], "category": category,
                         "response_parsed_boxed": parsed, "domain": case.metadata.get("domain")}
        if name in harness.LLM_EVAL_FUNCTIONS:
            if options.get("require_non_empty", True) and (
                not prediction.answer.strip() or not case.reference.strip() or not parsed.strip()
            ):
                correct = False
                self._details["judge_skipped"] = "empty answer or reference"
            else:
                judge = self._get_judge()
                correct = judge.score(name, question=case.request.question, reference=case.reference,
                                      raw=prediction.answer, parsed=parsed)
                self._details["judge"] = copy.deepcopy(judge.details)
        else:
            correct = metrics.score_to_bool(metrics.eval_from_spec(
                case.metadata["eval_function"], parsed, case.reference,
            ))
        scores = {"score": float(correct and not unknown), "is_unknown": float(unknown),
                  "is_abstention_problem": float(name == "llm_abstention_checker"),
                  "category_index": float(CATEGORIES.index(category))}
        timing = prediction.metadata.get("timings", prediction.metadata.get("longmemeval_v2", {}))
        for key in ("memory_query_duration_seconds", "reader_duration_seconds"):
            if key in timing:
                scores[key] = timing[key]
        return scores

    def aggregate(self, scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
        return aggregate_scores(scores)

    def evaluation_details(self) -> dict[str, Any]:
        return copy.deepcopy(self._details)

    def close(self) -> None:
        if self._judge is not None:
            self._judge.close()
            self._judge = None
