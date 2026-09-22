"""Experiment configuration and construction of explicitly registered components."""

from __future__ import annotations

import copy
import inspect
import json
import re
import tomllib
from dataclasses import dataclass, field
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar


class ConfigError(ValueError):
    """An invalid or unreadable experiment configuration."""


class PluginError(ValueError):
    """A registered component could not be resolved or constructed."""


@dataclass(frozen=True)
class PluginSpec:
    name: str
    type: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationBinding:
    generator: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComponentSpec(PluginSpec):
    generation: GenerationBinding | None = None


@dataclass(frozen=True)
class RunConfig:
    benchmarks: tuple[ComponentSpec, ...]
    memories: tuple[ComponentSpec, ...]
    agent: ComponentSpec
    generators: dict[str, PluginSpec]
    results_dir: Path
    artifacts_dir: Path
    snapshot: dict[str, Any]


def _table(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a TOML table")
    return value


def _keys(table: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = table.keys() - allowed
    if unknown:
        raise ConfigError(f"Unknown key(s) in {label}: {', '.join(sorted(unknown))}")


def _plugin(value: Any, label: str) -> PluginSpec:
    table = _table(value, label)
    _keys(table, {"type", "options", "name"}, label)
    component_type = table.get("type")
    if not isinstance(component_type, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", component_type
    ):
        raise ConfigError(f"{label}.type must be a registered name, such as 'synthetic' or 'verbatim'")
    name = table.get("name", component_type)
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ConfigError(f"{label}.name must contain letters, digits, underscores, or hyphens")
    return PluginSpec(name, component_type, _table(table.get("options", {}), f"{label}.options"))


def _component(value: Any, label: str) -> ComponentSpec:
    table = dict(_table(value, label))
    generation = table.pop("generation", None)
    spec = _plugin(table, label)
    if label == "benchmarks" and "evaluator" in spec.options:
        raise ConfigError("benchmarks.options.evaluator was removed; use generators.<name> and benchmarks.generation")
    binding = None
    if generation is not None:
        generation = _table(generation, f"{label}.generation")
        _keys(generation, {"generator", "settings"}, f"{label}.generation")
        reference = generation.get("generator")
        if not isinstance(reference, str) or not reference:
            raise ConfigError(f"{label}.generation.generator must name a generator")
        binding = GenerationBinding(reference, _table(generation.get("settings", {}), f"{label}.generation.settings"))
    return ComponentSpec(spec.name, spec.type, spec.options, binding)


def _plugins(value: Any, label: str) -> tuple[ComponentSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{label} must be a nonempty array of tables ([[{label}]])")
    specs = tuple(_component(item, label) for item in value)
    if len({spec.name for spec in specs}) != len(specs):
        raise ConfigError(f"{label} names must be unique")
    return specs


def load_config(path: Path | str) -> RunConfig:
    """Load an experiment. Filesystem paths are relative to the working directory."""
    try:
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc
    if "generation" in raw:
        raise ConfigError("Top-level [generation] was removed; use [generators.<name>] and [agent.generation]")
    _keys(raw, {"run", "agent", "generators", "benchmarks", "memories"}, "config")
    run = _table(raw.get("run", {}), "run")
    _keys(run, {"results_dir", "artifacts_dir"}, "run")
    paths = {}
    for key, default in (("results_dir", "results"), ("artifacts_dir", "artifacts")):
        value = run.get(key, default)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"run.{key} must be a nonempty path string")
        paths[key] = Path(value).resolve()
    if paths["results_dir"] == paths["artifacts_dir"]:
        raise ConfigError("run.results_dir and run.artifacts_dir must be distinct directories")
    generators = {}
    for name, value in _table(raw.get("generators"), "generators").items():
        table = _table(value, f"generators.{name}")
        _keys(table, {"type", "options"}, f"generators.{name}")
        generators[name] = _plugin({**table, "name": name}, f"generators.{name}")
    config = RunConfig(
        benchmarks=_plugins(raw.get("benchmarks"), "benchmarks"),
        memories=_plugins(raw.get("memories"), "memories"),
        agent=_component(raw.get("agent", {"type": "shared"}), "agent"),
        generators=generators,
        results_dir=paths["results_dir"],
        artifacts_dir=paths["artifacts_dir"],
        snapshot={},
    )
    if config.agent.generation is None:
        raise ConfigError("agent.generation is required; select a named generator")
    for component in (config.agent, *config.benchmarks, *config.memories):
        if component.generation and component.generation.generator not in generators:
            raise ConfigError(f"Unknown generator '{component.generation.generator}' for '{component.name}'")
    # Record effective defaults as well as explicitly supplied options.
    def spec_dict(spec: PluginSpec) -> dict[str, Any]:
        result = {"name": spec.name, "type": spec.type, "options": spec.options}
        if isinstance(spec, ComponentSpec) and spec.generation is not None:
            result["generation"] = {"generator": spec.generation.generator, "settings": spec.generation.settings}
        return result

    config.snapshot.update({
        "run": {key: str(value) for key, value in paths.items()},
        "benchmarks": [spec_dict(spec) for spec in config.benchmarks],
        "memories": [spec_dict(spec) for spec in config.memories],
        "agent": spec_dict(config.agent),
        "generators": {name: {"type": spec.type, "options": spec.options} for name, spec in generators.items()},
    })
    try:
        json.dumps(config.snapshot, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError("Config options must contain JSON-compatible values (no dates or NaN)") from exc
    return config


T = TypeVar("T")


def resolve_class(spec: PluginSpec, registry: Mapping[str, type[T]], base: type[T]) -> type[T]:
    """Resolve and validate a class without constructing expensive backends."""
    if spec.type not in registry:
        available = ", ".join(sorted(registry)) or "(none)"
        raise PluginError(
            f"Unknown {base.__name__} type '{spec.type}' for '{spec.name}'. Available types: {available}"
        )
    component_class = registry[spec.type]
    if not isinstance(component_class, type) or not issubclass(component_class, base):
        raise PluginError(f"Registered type '{spec.type}' must be a subclass of {base.__name__}")
    if inspect.isabstract(component_class):
        missing = ", ".join(sorted(component_class.__abstractmethods__))
        raise PluginError(f"Registered type '{spec.type}' is abstract; implement: {missing}")
    return component_class


def instantiate(spec: PluginSpec, registry: Mapping[str, type[T]], base: type[T]) -> T:
    """Pass independent constructor options to the selected concrete class."""
    component_class = resolve_class(spec, registry, base)
    try:
        return component_class(**copy.deepcopy(spec.options))
    except Exception as exc:
        raise PluginError(
            f"Cannot initialize {base.__name__} '{spec.name}' (type '{spec.type}') "
            f"with option keys {sorted(spec.options)}: {exc}"
        ) from exc
