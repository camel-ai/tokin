from importlib.metadata import version

from .chat_templates import ChatTemplate, ChatTemplateError, get_chat_template
from .rollout import Generation, Prompt, Rollout
from .session import Session

__version__ = version("tokin")
__all__ = [
    "ChatTemplate",
    "ChatTemplateError",
    "Generation",
    "Prompt",
    "Rollout",
    "Session",
    "__version__",
    "get_chat_template",
]
