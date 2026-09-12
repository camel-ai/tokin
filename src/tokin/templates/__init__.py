from ..template import ChatTemplate
from .glm import GLMChatTemplate
from .qwen import QwenChatTemplate

TEMPLATES: tuple[type[ChatTemplate], ...] = (QwenChatTemplate, GLMChatTemplate)


def get(name: str) -> type[ChatTemplate]:
    """The family called `name`, or the one a checkpoint id in `models` belongs to."""
    for template in TEMPLATES:
        if name == template.name or name in template.models:
            return template
    raise ValueError(f"unknown chat template {name!r}; known: {[t.name for t in TEMPLATES]}")


__all__ = ["GLM", "TEMPLATES", "Qwen", "get"]
