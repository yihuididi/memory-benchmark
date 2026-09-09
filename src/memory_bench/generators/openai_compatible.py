"""Generator for user-supplied OpenAI-compatible model servers."""

from __future__ import annotations

import base64
import copy
import mimetypes
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from memory_bench.generators import register_generator
from memory_bench.generators.base import BaseGenerator, UnsupportedModelAdapterError
from memory_bench.types import Message, ModelAdapterRef, Prediction


def create_client(**kwargs: Any):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Model endpoints require: uv sync --extra model-api") from exc
    return OpenAI(max_retries=2, **kwargs)


def _image(path: str) -> dict[str, Any]:
    file = Path(path)
    mime = mimetypes.guess_type(str(file))[0] or "image/png"
    data = base64.b64encode(file.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def encode_messages(messages: Sequence[Message], attachments: Sequence[str] = ()) -> list[dict[str, Any]]:
    encoded = []
    for message in messages:
        if isinstance(message.content, str):
            content = message.content
        else:
            content = []
            for block in message.content:
                if block.get("type") == "image_path":
                    content.append(_image(block["image_path"]))
                elif block.get("type") in ("text", "image_url"):
                    content.append(copy.deepcopy(block))
                else:
                    raise ValueError(f"Unsupported message block: {block.get('type')!r}")
        encoded.append({"role": message.role, "content": content})
    if attachments:
        last_user = next((item for item in reversed(encoded) if item["role"] == "user"), None)
        if last_user is None:
            raise ValueError("Image attachments require a user message")
        if isinstance(last_user["content"], str):
            last_user["content"] = [{"type": "text", "text": last_user["content"]}]
        last_user["content"].extend(_image(path) for path in attachments)
    return encoded


@register_generator("openai_compatible")
class OpenAICompatibleGenerator(BaseGenerator):
    def __init__(self, *, model: str, base_url: str, api_key_env: str = "MODEL_API_KEY",
                 timeout_seconds: float = 600.0) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an explicit HTTP(S) model endpoint")
        self.model = model
        self.client = create_client(base_url=base_url, api_key=os.environ.get(api_key_env) or "EMPTY",
                                    timeout=timeout_seconds)

    def generate(self, messages: Sequence[Message], *, settings: Mapping[str, Any],
                 model_adapter: ModelAdapterRef | None = None,
                 attachments: Sequence[str] = ()) -> Prediction:
        if model_adapter is not None:
            raise UnsupportedModelAdapterError("This endpoint generator cannot load model adapters")
        allowed = {"max_tokens", "max_completion_tokens", "temperature", "top_p", "stop", "seed",
                   "frequency_penalty", "presence_penalty", "reasoning_effort", "extra_body"}
        if set(settings) - allowed:
            raise ValueError(f"Unsupported generation settings: {sorted(set(settings) - allowed)}")
        if "max_tokens" in settings and "max_completion_tokens" in settings:
            raise ValueError("Use only one of max_tokens and max_completion_tokens")
        # extra_body is merged at the wire boundary by the SDK; protect these fields too.
        extra = settings.get("extra_body", {})
        if not isinstance(extra, Mapping) or set(extra) & {"model", "messages", "stream", "n"}:
            raise ValueError("Unsupported generation settings in extra_body")
        response = self.client.chat.completions.create(
            model=self.model, messages=encode_messages(messages, attachments), **copy.deepcopy(dict(settings)),
        )
        choice = response.choices[0]
        answer = choice.message.content
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Generator returned no answer content")
        if choice.finish_reason not in ("stop", "length"):
            raise ValueError(f"Generator failed to produce an answer: {choice.finish_reason}")
        usage = getattr(response, "usage", None)
        return Prediction(answer, {"model": self.model, "finish_reason": choice.finish_reason,
                                   "usage": {key: int(getattr(usage, key, 0) or 0) for key in
                                             ("prompt_tokens", "completion_tokens", "total_tokens")}})

    def close(self) -> None:
        self.client.close()
