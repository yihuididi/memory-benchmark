from dataclasses import replace
import json

import pytest

from memory_bench.config import ConfigError, GenerationBinding, PluginSpec, load_config
from memory_bench.generators import BaseGenerator
from memory_bench.generators.lazy import LazyGenerator
from memory_bench.memories import BaseMemory
from memory_bench.runner import run
from memory_bench.types import MemoryAugmentation, Message, Prediction


CONFIG = '''
[generators.shared]
type = "deterministic"
[agent]
type = "shared"
[agent.generation]
generator = "shared"
[agent.generation.settings]
temperature = 0.6
[[benchmarks]]
type = "synthetic"
[[memories]]
type = "none"
'''


def load(tmp_path, text):
    path = tmp_path / "run.toml"
    path.write_text(text)
    return load_config(path)


def test_optional_bindings_and_snapshot(tmp_path):
    config = load(tmp_path, CONFIG)
    assert config.memories[0].generation is config.benchmarks[0].generation is None
    assert config.snapshot["agent"]["generation"] == {"generator": "shared", "settings": {"temperature": .6}}
    assert config.snapshot["generators"]["shared"] == {"type": "deterministic", "options": {}}
    assert "generation" not in config.snapshot


@pytest.mark.parametrize("text,match", [
    (CONFIG.replace('generator = "shared"', 'generator = "missing"'), "Unknown generator"),
    (CONFIG.replace('[agent.generation]\ngenerator = "shared"\n[agent.generation.settings]\ntemperature = 0.6\n', ''), "agent.generation is required"),
    (CONFIG.replace('generator = "shared"', 'generator = 3'), "must name a generator"),
    (CONFIG.replace('generator = "shared"', 'generator = "shared"\nsettings = 3').replace('[agent.generation.settings]\ntemperature = 0.6\n', ''), "must be a TOML table"),
    (CONFIG + '\n[generation]\ntype = "deterministic"', "was removed"),
    (CONFIG.replace('type = "synthetic"', 'type = "synthetic"\n[benchmarks.options.evaluator]\nbackend = "transformers"'), "evaluator was removed"),
    (CONFIG.replace('[generators.shared]', '[generators.shared]\nsettings = {}'), "Unknown key"),
])
def test_invalid_configuration(tmp_path, text, match):
    with pytest.raises(ConfigError, match=match):
        load(tmp_path, text)


def test_shared_definition_has_independent_component_settings_and_episode_state(config_factory, install_plugins):
    backends, memories = [], []
    class Generator(BaseGenerator):
        def __init__(self):
            self.calls = []
            self.closed = 0
            backends.append(self)
        def generate(self, messages, *, settings, **kwargs):
            self.calls.append((messages, dict(settings)))
            settings["mutated"] = True
            return Prediction("ok")
        def close(self):
            self.closed += 1
    class Memory(BaseMemory):
        def __init__(self):
            memories.append(self)
        def prepare(self, records, workspace):
            self.generation.generate([Message("user", "prepare")], settings=self.generation_settings)
        def augment(self, request):
            self.generation.generate([Message("user", "retrieve")], settings=self.generation_settings)
            return MemoryAugmentation()
        def close(self):
            assert not self.generation.closed
    registered = install_plugins(Generation=Generator, Memory=Memory)
    config = config_factory(generation=registered["Generation"], memories=[("memory", registered["Memory"], {})])
    config = replace(config, memories=(replace(config.memories[0], generation=GenerationBinding("reader", {"temperature": .2})),))
    output = run(config)
    assert json.loads((output / "summary.json").read_text())["status"] == "completed"
    assert len(backends) == 4  # Agent and memory for each of two episodes.
    assert len(memories) == 2
    assert all(backend.closed == 1 for backend in backends)
    assert sorted(len(backend.calls) for backend in backends) == [2, 2, 3, 3]
    for backend in backends:
        assert len({call[1]["temperature"] for call in backend.calls}) == 1
        assert all("mutated" not in call[1] for call in backend.calls)
    assert all(memory.generation_settings == {"temperature": .2} for memory in memories)


def test_lazy_unused_handle_never_constructs_backend(install_plugins):
    class Generator(BaseGenerator):
        def __init__(self):
            pytest.fail("Unused backend must stay lazy")
        def generate(self, *args, **kwargs):
            pass
        def close(self):
            pytest.fail("Unconstructed backend must not be closed")
    registered = install_plugins(Generation=Generator)
    handle = LazyGenerator(PluginSpec("unused", registered["Generation"]))
    handle.close()
    handle.close()


def test_memory_preparation_failure_closes_borrowed_generator(config_factory, install_plugins):
    closed = []
    class Generator(BaseGenerator):
        def generate(self, *args, **kwargs):
            return Prediction("ok")
        def close(self):
            closed.append("generator")
    class Memory(BaseMemory):
        def prepare(self, records, workspace):
            self.generation.generate([], settings={})
            raise RuntimeError("prepare failed")
        def augment(self, request):
            pytest.fail("Must not retrieve")
        def close(self):
            closed.append("memory")
    registered = install_plugins(Generation=Generator, Memory=Memory)
    config = config_factory(generation=registered["Generation"], memories=[("memory", registered["Memory"], {})])
    config = replace(config, memories=(replace(config.memories[0], generation=GenerationBinding("reader")),))
    with pytest.raises(RuntimeError, match="prepare failed"):
        run(config)
    assert closed == ["memory", "generator"]


def test_openai_preflight_and_close_do_not_create_client(monkeypatch):
    from memory_bench.generators.openai_compatible import OpenAICompatibleGenerator
    def unexpected(**kwargs):
        pytest.fail("Preflight must not create a client")
    monkeypatch.setattr("memory_bench.generators.openai_compatible.create_client", unexpected)
    generator = OpenAICompatibleGenerator(model="test", base_url="http://localhost:1234/v1")
    generator.preflight()
    generator.close()


def test_binding_failure_closes_component_and_constructed_backend(config_factory, install_plugins):
    closed = []
    class Generator(BaseGenerator):
        def generate(self, *args, **kwargs):
            return Prediction("ok")
        def close(self):
            closed.append("generator")
    class Memory(BaseMemory):
        def bind_generation(self, generator, settings):
            super().bind_generation(generator, settings)
            generator.preflight()
            raise RuntimeError("binding failed")
        def prepare(self, *args):
            pytest.fail("Must not prepare")
        def augment(self, request):
            pytest.fail("Must not retrieve")
        def close(self):
            assert not self.generation.closed
            closed.append("memory")
    registered = install_plugins(Generation=Generator, Memory=Memory)
    config = config_factory(generation=registered["Generation"], memories=[("memory", registered["Memory"], {})])
    config = replace(config, memories=(replace(config.memories[0], generation=GenerationBinding("reader")),))
    with pytest.raises(RuntimeError, match="binding failed"):
        run(config)
    assert closed == ["memory", "generator"]
