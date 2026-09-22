"""Runner-owned backend handles; definitions are shared, instances are not."""
import copy

from memory_bench.config import instantiate
from memory_bench.generators import GENERATORS
from memory_bench.generators.base import BaseGenerator


class LazyGenerator(BaseGenerator):
    def __init__(self, spec):
        self.spec = spec
        self.backend = None
        self.closed = False

    def _get(self):
        if self.closed:
            raise RuntimeError("Generator handle is closed")
        if self.backend is None:
            self.backend = instantiate(self.spec, GENERATORS, BaseGenerator)
        return self.backend

    def preflight(self):
        self._get().preflight()

    def _describe(self, prediction, settings):
        from dataclasses import replace
        return replace(prediction, metadata={**prediction.metadata,
            "backend": self.spec.type, "generator": self.spec.name,
            "options": copy.deepcopy(self.spec.options), "settings": copy.deepcopy(dict(settings))})

    def generate(self, messages, *, settings, model_adapter=None, attachments=()):
        return self._describe(self._get().generate(messages, settings=copy.deepcopy(dict(settings)),
            model_adapter=model_adapter, attachments=attachments), settings)

    def generate_structured(self, messages, *, settings, schema):
        return self._describe(self._get().generate_structured(messages,
            settings=copy.deepcopy(dict(settings)), schema=schema), settings)

    def close(self):
        if not self.closed:
            self.closed = True
            if self.backend is not None:
                self.backend.close()
