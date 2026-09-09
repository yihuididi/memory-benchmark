"""Local text judge: Transformers inference constrained by Outlines, then Pydantic."""

from __future__ import annotations

import gc
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from memory_bench.benchmarks.longmemeval_v2 import judge_messages


class BinaryJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    label: int = Field(ge=0, le=1)
    reason: str = Field(min_length=1)


class LocalJudge:
    def __init__(self, *, model_path: str, device: str = "cuda", dtype: str = "auto",
                 max_new_tokens: int = 512, max_context_tokens: int | None = None,
                 chat_template_kwargs: dict[str, Any] | None = None) -> None:
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("evaluator.model_path must be a nonempty local path")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("evaluator.device must be a nonempty device string")
        if dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("evaluator.dtype must be auto, float32, float16, or bfloat16")
        for key, value in (("max_new_tokens", max_new_tokens), ("max_context_tokens", max_context_tokens)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{key} must be a positive integer")
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be a positive integer")
        self.path = Path(model_path).expanduser().resolve()
        self.device, self.dtype = device, dtype
        self.max_new_tokens, self.max_context_tokens = max_new_tokens, max_context_tokens
        # Qwen3 otherwise opens a thinking block, which conflicts with JSON-only decoding.
        self.chat_template_kwargs = {"enable_thinking": False, **(chat_template_kwargs or {})}
        reserved = {"tokenize", "add_generation_prompt", "conversation", "return_tensors"}
        if reserved & self.chat_template_kwargs.keys():
            raise ValueError("chat_template_kwargs cannot override prompt encoding options")
        self.model = self.tokenizer = self.generator = None
        self.details: dict[str, Any] = {}

    def preflight(self) -> None:
        if not (self.path / "config.json").is_file():
            raise FileNotFoundError(f"Local judge model missing: {self.path}/config.json")
        try:
            import outlines
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError("Local judge requires: uv sync --extra local-judge") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Local judge requested CUDA, but CUDA is unavailable. Fix the GPU driver "
                               "or explicitly set evaluator.device = 'cpu' (slower).")

    def _load(self) -> None:
        if self.generator is not None:
            return
        self.preflight()
        import outlines
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True,
                                                       trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError(f"Local judge requires an instruction model with a chat template: {self.path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.path), local_files_only=True, trust_remote_code=False,
            dtype=self.dtype, device_map=self.device,
        ).eval()
        self.generator = outlines.Generator(outlines.from_transformers(self.model, self.tokenizer),
                                            BinaryJudgement)

    def score(self, name: str, **inputs: str) -> bool:
        self.details = {}
        self._load()
        prompt = self.tokenizer.apply_chat_template(
            judge_messages(name, **inputs), tokenize=False, add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        # Match the default special-token behavior of Outlines' Transformers tokenizer.
        input_tokens = len(self.tokenizer.encode(prompt))
        limits = [getattr(self.model.config, "max_position_embeddings", None),
                  self.tokenizer.model_max_length, self.max_context_tokens]
        limits = [x for x in limits if isinstance(x, int) and 0 < x < 10**12]
        if not limits:
            raise ValueError("Cannot determine judge context window; configure max_context_tokens")
        context_limit = min(limits)
        if input_tokens + self.max_new_tokens > context_limit:
            raise ValueError(f"Judge prompt ({input_tokens} tokens) plus output allowance "
                             f"({self.max_new_tokens}) exceeds context limit {context_limit}; "
                             "use a larger-context evaluator. Judge inputs are never truncated.")
        import torch
        with torch.inference_mode():
            raw = self.generator(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)
        try:
            judgement = BinaryJudgement.model_validate_json(raw)
        except ValueError as exc:
            raise ValueError("Local judge returned invalid/incomplete JSON; increase max_new_tokens "
                             "or inspect the evaluator model. No score was recorded.") from exc
        self.details = {
            "backend": "transformers", "model_path": str(self.path),
            "device": self.device, "dtype": str(self.model.dtype),
            "max_new_tokens": self.max_new_tokens, "max_context_tokens": context_limit,
            "chat_template_kwargs": self.chat_template_kwargs,
            "do_sample": False, "input_tokens": input_tokens,
            "versions": {name: version(name) for name in ("transformers", "outlines", "torch", "pydantic")},
            **judgement.model_dump(), "raw": raw,
        }
        return judgement.label == 1

    def close(self) -> None:
        self.generator = self.model = self.tokenizer = None
        gc.collect()
        if self.device.startswith("cuda"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
