"""Local text inference with optional Outlines-constrained JSON output."""
from __future__ import annotations

import copy
import gc
from importlib.metadata import version
from pathlib import Path

from memory_bench.generators import register_generator
from memory_bench.generators.base import BaseGenerator, UnsupportedModelAdapterError
from memory_bench.types import Prediction


@register_generator("transformers")
class TransformersGenerator(BaseGenerator):
    def __init__(self, *, model_path: str, device: str = "cuda", dtype: str = "auto",
                 max_context_tokens: int | None = None):
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("model_path must be a nonempty local path")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a nonempty device string")
        if dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")
        if max_context_tokens is not None and (type(max_context_tokens) is not int or max_context_tokens <= 0):
            raise ValueError("max_context_tokens must be a positive integer")
        self.path = Path(model_path).expanduser().resolve()
        self.device, self.dtype = device, dtype
        self.max_context_tokens = max_context_tokens
        self.model = self.tokenizer = self.wrapped = None

    def preflight(self):
        if not (self.path / "config.json").is_file():
            raise FileNotFoundError(f"Local model missing: {self.path}/config.json")
        try:
            import outlines
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError("Local generation requires: uv sync --extra local-transformers") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Local generation requested CUDA, but CUDA is unavailable; configure device='cpu' or fix the GPU driver")

    def _load(self):
        if self.wrapped is not None:
            return
        self.preflight()
        import outlines
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError(f"Local generation requires an instruction model with a chat template: {self.path}")
        self.model = AutoModelForCausalLM.from_pretrained(str(self.path), local_files_only=True,
            trust_remote_code=False, dtype=self.dtype, device_map=self.device).eval()
        self.wrapped = outlines.from_transformers(self.model, self.tokenizer)

    def generate(self, messages, *, settings, model_adapter=None, attachments=()):
        if model_adapter is not None:
            raise UnsupportedModelAdapterError("Local text generation cannot load model adapters")
        if attachments:
            raise ValueError("Local text generation does not support attachments")
        return self._generate(messages, settings=settings, schema=None)

    def generate_structured(self, messages, *, settings, schema):
        return self._generate(messages, settings=settings, schema=schema)

    def _generate(self, messages, *, settings, schema):
        options = copy.deepcopy(dict(settings))
        allowed = {"max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "chat_template_kwargs"}
        if options.keys() - allowed:
            raise ValueError(f"Unsupported generation settings: {sorted(options.keys() - allowed)}")
        options.setdefault("max_new_tokens", 512)
        options.setdefault("do_sample", False)
        if type(options["max_new_tokens"]) is not int or options["max_new_tokens"] <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(options["do_sample"]) is not bool:
            raise ValueError("do_sample must be a boolean")
        template = options.pop("chat_template_kwargs", {})
        if not isinstance(template, dict) or {"tokenize", "add_generation_prompt", "conversation", "return_tensors"} & template.keys():
            raise ValueError("chat_template_kwargs cannot override prompt encoding options")
        if any(not isinstance(message.content, str) for message in messages):
            raise ValueError("Local text generation only supports text messages")
        self._load()
        prompt = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False, add_generation_prompt=True, **template)
        input_tokens = len(self.tokenizer.encode(prompt))
        limits = [getattr(self.model.config, "max_position_embeddings", None),
                  self.tokenizer.model_max_length, self.max_context_tokens]
        limits = [x for x in limits if isinstance(x, int) and 0 < x < 10**12]
        if not limits:
            raise ValueError("Cannot determine context window; configure max_context_tokens")
        context_limit = min(limits)
        if input_tokens + options["max_new_tokens"] > context_limit:
            raise ValueError(f"Prompt ({input_tokens} tokens) plus output allowance ({options['max_new_tokens']}) "
                             f"exceeds context limit {context_limit}; inputs are never truncated")
        import outlines
        import torch
        generator = outlines.Generator(self.wrapped, schema)
        with torch.inference_mode():
            raw = generator(prompt, **options)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Generator returned no answer content")
        return Prediction(raw, {
            "backend": "transformers", "model_path": str(self.path), "device": self.device,
            "dtype": str(self.model.dtype), **options, "max_context_tokens": context_limit,
            "chat_template_kwargs": template, "input_tokens": input_tokens,
            "versions": {name: version(name) for name in ("transformers", "outlines", "torch")},
        })

    def close(self):
        loaded = self.model is not None
        self.wrapped = self.model = self.tokenizer = None
        if loaded:
            gc.collect()
            if self.device.startswith("cuda"):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
