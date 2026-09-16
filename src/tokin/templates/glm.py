from ..template import ChatTemplate


class GLMChatTemplate(ChatTemplate):
    name = "glm"
    reasoning_start = "<think>"
    reasoning_end = "</think>"
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
        "zai-org/GLM-5.3",
        "zai-org/GLM-5.3-Flash",
    )
