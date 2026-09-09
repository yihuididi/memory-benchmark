"""Registration decorators shared by the component packages."""

import inspect
import re
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def make_register(
    registry: dict[str, type[T]], base: type[T]
) -> Callable[[str], Callable[[type[T]], type[T]]]:
    """Create a decorator that registers concrete subclasses without instantiating them."""
    def register(name: str) -> Callable[[type[T]], type[T]]:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("Registration name must contain letters, digits, underscores, or hyphens")

        def decorate(component_class: type[T]) -> type[T]:
            if not isinstance(component_class, type) or not issubclass(component_class, base):
                raise TypeError(f"Registered type '{name}' must be a subclass of {base.__name__}")
            if inspect.isabstract(component_class):
                missing = ", ".join(sorted(component_class.__abstractmethods__))
                raise TypeError(f"Registered type '{name}' is abstract; implement: {missing}")
            if name in registry:
                raise ValueError(f"{base.__name__} type '{name}' is already registered")
            registry[name] = component_class
            return component_class

        return decorate

    return register
