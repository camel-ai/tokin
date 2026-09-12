from ..template import ChatTemplate


class QwenChatTemplate(ChatTemplate):
    name = "qwen"
    turn_end = "<|im_end|>\n"
    stop = ("<|im_end|>", "<|endoftext|>")
    kwargs = ("enable_thinking", "preserve_thinking", "add_vision_id")
    models = (
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/QwQ-32B",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.6-27B",
    )
