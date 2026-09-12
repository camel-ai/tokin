from collections.abc import Iterable

from ..template import ChatTemplate
from ..types import PromptMessage, ToolCall, ToolSchema


class GLMChatTemplate(ChatTemplate):
    name = "glm"
    stop = ("<|user|>", "<|observation|>", "<|endoftext|>")
    kwargs = ("enable_thinking", "clear_thinking", "reasoning_effort")
    models = (
        "zai-org/GLM-4.5",
        "zai-org/GLM-4.5-Air",
        "zai-org/GLM-4.6",
        "zai-org/GLM-4.7",
        "zai-org/GLM-4.7-Flash",
        "zai-org/GLM-5",
        "zai-org/GLM-5.1",
        "zai-org/GLM-5.2",
    )

    def apply_increment(
        self,
        messages: list[PromptMessage],
        tools: list[ToolSchema] | None = None,
        *,
        tool_calls: Iterable[ToolCall] | None = None,
    ) -> str:
        """Drop the opening role tag: the model already produced it as its stop token."""
        text = super().apply_increment(messages, tools, tool_calls=tool_calls)
        for opener in ("<|user|>", "<|observation|>", "<|system|>"):
            if text.startswith(opener):
                return text[len(opener) :]
        return text
