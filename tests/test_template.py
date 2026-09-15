from __future__ import annotations

import jinja2
import pytest

from tokin.template import ChatTemplate, TemplateError
from tokin.templates import get

# One private-use character per control token, so each is a single id.
START, END = "", ""
CHATML = (
    "{%- for m in messages %}{{ start ~ m.role ~ '\\n' }}"
    "{%- if m.role == 'system' and tools %}tools={{ tools | tojson }}{% endif %}"
    "{{ m.content }}"
    "{%- for c in m.tool_calls or [] %}<call>{{ c.function.arguments | tojson }}</call>{% endfor %}"
    "{%- if m.tool_call_id %}<id>{{ m.tool_call_id }}</id>{% endif %}"
    "{{ end ~ '\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ start ~ 'assistant\\n' }}"
    "{%- if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}{% endif %}"
)

USER = {"role": "user", "content": "hi"}
SYSTEM = {"role": "system", "content": "be terse"}
TOOLS = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
CALLS = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}]
TOOL = {"role": "tool", "tool_call_id": "c1", "content": "18C"}


class FakeTokenizer:
    def __init__(self, template: str = CHATML) -> None:
        self.chat_template = template
        self._env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])

    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt, tokenize, **kwargs):
        def raise_exception(message):
            raise jinja2.TemplateError(message)

        return self._env.from_string(self.chat_template).render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            raise_exception=raise_exception,
            start=START,
            end=END,
            **kwargs,
        )

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class ChatML(ChatTemplate):
    name = "chatml"
    turn_end = f"{END}\n"
    stop = (END,)
    kwargs = ("enable_thinking",)


def test_conform_parses_string_arguments_without_touching_the_input():
    assistant = {"role": "assistant", "content": "", "tool_calls": CALLS}
    conformed = ChatML(FakeTokenizer()).conform([USER, assistant])
    assert conformed[1]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert CALLS[0]["function"]["arguments"] == '{"x": 1}'
    assert conformed[0] == USER


def test_apply_is_the_tokenizers_own_render():
    tokenizer = FakeTokenizer()
    assert ChatML(tokenizer).apply([SYSTEM, USER]) == tokenizer.apply_chat_template(
        [SYSTEM, USER], add_generation_prompt=True, tokenize=False
    )


def test_kwargs_reach_the_template():
    assert ChatML(FakeTokenizer(), enable_thinking=False).apply([USER]).endswith("<think></think>")


def test_unknown_kwarg_is_rejected():
    with pytest.raises(ValueError, match="enable_thinkin"):
        ChatML(FakeTokenizer(), enable_thinkin=False)


def test_stop_tokens_must_be_single_ids():
    with pytest.raises(ValueError, match="not one"):
        get("qwen")(FakeTokenizer())


class TestApplyIncrement:
    def test_tool_results_then_user(self):
        got = ChatML(FakeTokenizer()).apply_increment([TOOL, USER], TOOLS, tool_calls=CALLS)
        assert got == f"\n{START}tool\n18C<id>c1</id>{END}\n{START}user\nhi{END}\n{START}assistant\n"

    def test_tool_results_need_their_calls(self):
        with pytest.raises(TemplateError, match="tool_calls"):
            ChatML(FakeTokenizer()).apply_increment([TOOL], TOOLS)

    def test_tool_results_come_first(self):
        with pytest.raises(TemplateError, match="before"):
            ChatML(FakeTokenizer()).apply_increment([USER, TOOL], TOOLS, tool_calls=CALLS)

    def test_opening_token_must_be_a_stop(self):
        class NoTurnEnd(ChatML):
            turn_end = ""

        with pytest.raises(TemplateError, match="never stops on"):
            NoTurnEnd(FakeTokenizer()).apply_increment([USER])


class TestApplyAfterStub:
    def test_template_error_is_wrapped(self):
        strict = CHATML.replace(
            "{%- for m in messages %}",
            "{%- for m in messages %}{% if m.role == 'system' and not loop.first %}{{ raise_exception('system first') }}{% endif %}",
        )
        with pytest.raises(TemplateError, match="system first"):
            ChatML(FakeTokenizer(strict)).apply_increment([SYSTEM])

    def test_rewriting_earlier_turns_is_refused(self):
        hoisting = "{%- for m in messages if m.role == 'system' %}[{{ m.content }}]{% endfor %}" + CHATML.replace(
            "{%- for m in messages %}", "{%- for m in messages if m.role != 'system' %}"
        )
        with pytest.raises(TemplateError, match="rewrites"):
            ChatML(FakeTokenizer(hoisting)).apply_increment([SYSTEM])


def test_get_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown chat template"):
        get("gpt2")
