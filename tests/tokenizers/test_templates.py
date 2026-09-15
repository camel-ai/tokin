import json

import jinja2
import pytest
from jinja2 import meta
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from tokin.template import TemplateError
from tokin.templates import TEMPLATES, GLMChatTemplate, QwenChatTemplate, get

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "d",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]
SYSTEM = {"role": "system", "content": "You are terse."}
USER = {"role": "user", "content": "Weather in Paris?"}
REPLY = {"role": "assistant", "content": "Sunny.", "reasoning_content": "hmm"}
CALLS = [
    {"id": f"call_{i}", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": c})}}
    for i, c in enumerate(("Paris", "London"), 1)
]
CALL = {"role": "assistant", "content": "", "reasoning_content": "hmm", "tool_calls": CALLS[:1]}
CALL2 = {**CALL, "tool_calls": CALLS}
RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "18C"}
RESULT2 = {"role": "tool", "tool_call_id": "call_2", "content": "12C"}
CONTENTS = ["And London?", "旧金山呢？", "  spaced out \n\n", "", '{"json": true}', "a <|im_end|> inside"]


def user(content):
    return {"role": "user", "content": content}


# history, new messages, tools, the calls the results answer
SCENARIOS = {
    **{f"user:{c!r}": ([SYSTEM, USER, REPLY], [user(c)], None, None) for c in CONTENTS},
    "tool": ([SYSTEM, USER, CALL], [RESULT], TOOLS, CALLS[:1]),
    "tool,user": ([SYSTEM, USER, CALL], [RESULT, user("And London?")], TOOLS, CALLS[:1]),
    "tool,tool": ([SYSTEM, USER, CALL2], [RESULT, RESULT2], TOOLS, CALLS),
    "system,user": (
        [SYSTEM, USER, REPLY],
        [{"role": "system", "content": "Answer in French."}, user("Merci?")],
        None,
        None,
    ),
}
# Qwen3.5 raises on any system message that is not the first; GLM never stops on <|system|>. Both fork.
REFUSES_MID_SYSTEM = {"Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-27B", *GLMChatTemplate.models}


def test_apply_matches_the_tokenizer(template):
    conversation = [SYSTEM, USER, CALL, RESULT]
    expected = template.tokenizer.apply_chat_template(
        template.conform(conversation), tools=TOOLS, add_generation_prompt=True, tokenize=True, return_dict=False
    )
    assert template.encode(template.apply(conversation, TOOLS)) == expected


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_increment_is_a_token_suffix_of_the_full_render(template, scenario):
    history, new, tools, tool_calls = scenario
    if new[0]["role"] == "system" and template.tokenizer.name_or_path in REFUSES_MID_SYSTEM:
        with pytest.raises(TemplateError):
            template.apply_increment(new, tools, tool_calls=tool_calls)
        return
    increment = template.encode(template.apply_increment(new, tools, tool_calls=tool_calls))
    full = template.encode(template.apply(history + new, tools))
    prefix = full[: len(full) - len(increment)]
    # The increment begins right after the model's stop token, so nothing is missing or doubled at the seam.
    assert full == prefix + increment and prefix[-1] in template.stop_ids


def test_suffix_check_fails_without_the_turn_end_remainder():
    # Negative control: drop the "\n" Qwen's model never writes and the property must break.
    template = QwenChatTemplate(AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"))
    increment = template.encode(template.apply_increment([user("And London?")]))[1:]
    full = template.encode(template.apply([SYSTEM, USER, REPLY, user("And London?")]))
    prefix = full[: len(full) - len(increment)]
    assert prefix[-1] not in template.stop_ids


def test_declared_eos_are_stop_tokens(template):
    model = template.tokenizer.name_or_path
    try:
        eos = GenerationConfig.from_pretrained(model).eos_token_id
    except OSError:  # no generation_config.json; the model config still names one
        eos = AutoConfig.from_pretrained(model).get_text_config().eos_token_id
    assert set(eos if isinstance(eos, list) else [eos]) <= template.stop_ids


def test_get_returns_the_family_of_each_model(template):
    assert get(template.tokenizer.name_or_path) is type(template)


@pytest.mark.parametrize("family", TEMPLATES, ids=lambda t: t.name)
def test_declared_kwargs_are_read_by_some_template_of_the_family(family):
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    read = set()
    for model in family.models:
        read |= meta.find_undeclared_variables(env.parse(AutoTokenizer.from_pretrained(model).chat_template))
    assert set(family.kwargs) <= read
