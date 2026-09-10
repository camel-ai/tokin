from __future__ import annotations

from openai.types.chat import (
    ChatCompletionAssistantMessageParam as _AssistantMessage,
    ChatCompletionDeveloperMessageParam as DeveloperMessage,
    ChatCompletionMessageToolCallUnionParam as ToolCall,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionToolMessageParam as ToolMessage,
    ChatCompletionToolParam as ToolSchema,
    ChatCompletionUserMessageParam as UserMessage,
)


class AssistantMessage(_AssistantMessage, total=False):
    """OpenAI's assistant message plus the field reasoning models keep their thinking in."""

    reasoning_content: str


type PromptMessage = DeveloperMessage | SystemMessage | UserMessage | ToolMessage
type Message = PromptMessage | AssistantMessage

__all__ = [
    "AssistantMessage",
    "DeveloperMessage",
    "Message",
    "PromptMessage",
    "SystemMessage",
    "ToolCall",
    "ToolMessage",
    "ToolSchema",
    "UserMessage",
]
