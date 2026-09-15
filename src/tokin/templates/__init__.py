import importlib

from ..template import ChatTemplate
from .glm import GLMChatTemplate
from .qwen import QwenChatTemplate

TEMPLATES: dict[str, type[ChatTemplate]] = {t.name: t for t in (QwenChatTemplate, GLMChatTemplate)}


def get_template(name: str) -> type[ChatTemplate]:
    """The family called `name`, the one a checkpoint id in `models` belongs to, or a `module:Class` to import."""
    if name in TEMPLATES:
        return TEMPLATES[name]
    for template in TEMPLATES.values():
        if name in template.models:
            return template
    if ":" in name:
        module, _, attr = name.partition(":")
        template = getattr(importlib.import_module(module), attr)
        if isinstance(template, type) and issubclass(template, ChatTemplate):
            return template
        raise TypeError(f"{name} is not a ChatTemplate subclass")
    raise ValueError(f"unknown chat template {name!r}; known: {list(TEMPLATES)}")


__all__ = ["TEMPLATES", "get_template"]
