from ..template import ChatTemplate


class QwenChatTemplate(ChatTemplate):
    """Qwen2.5 through Qwen3, which write tool calls as JSON inside `<tool_call>`."""

    name = "qwen"
    turn_end = "<|im_end|>\n"
    stop = ("<|im_end|>", "<|endoftext|>")
    kwargs = ("enable_thinking",)
    models = (
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-72B",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
    )


class Qwen35ChatTemplate(ChatTemplate):
    """Qwen3.5 onwards, which write tool calls as `<function=…>` XML inside `<tool_call>`."""

    name = "qwen3.5"
    turn_end = "<|im_end|>\n"
    stop = ("<|im_end|>", "<|endoftext|>")
    kwargs = ("enable_thinking", "preserve_thinking", "add_vision_id")
    models = (
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.8-2.4T-A95B",
    )
