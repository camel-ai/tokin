import pytest

from tokin.render import Hooks, Renderer, RenderError


class FakeTokenizer:
    """Renders each message as `<role>content|`, so deltas are readable by eye.

    A terminal assistant turn also gets a `~` after the `|`, standing in for the
    newline real templates put between messages — that is what the separator
    probe has to find.
    """

    def apply_chat_template(self, conversation, *, tokenize=False, add_generation_prompt=True, tools=None, **kw):
        parts = []
        if tools:
            parts.append(f"<tools:{len(tools)}|")
        for i, m in enumerate(conversation):
            parts.append(f"<{m['role']}>{m.get('content') or ''}|")
            last_assistant = m["role"] == "assistant" and i == len(conversation) - 1
            if last_assistant and not add_generation_prompt:
                parts.append("~")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def encode(self, text, *, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "".join(chr(i) for i in token_ids)


def ids(text):
    return [ord(c) for c in text]


def test_render_produces_the_whole_prompt():
    r = Renderer(FakeTokenizer())
    assert r.render([{"role": "user", "content": "hi"}]) == ids("<user>hi|<assistant>")


def test_delta_frames_appended_messages_as_a_continuation():
    r = Renderer(FakeTokenizer())
    # The fake prefix cancels, leaving the new turn plus the assistant opener.
    assert r.delta([{"role": "user", "content": "next"}]) == ids("<user>next|<assistant>")


def test_delta_of_nothing_is_empty():
    assert Renderer(FakeTokenizer()).delta([]) == []


def test_delta_carries_tool_results():
    r = Renderer(FakeTokenizer())
    got = r.decode(r.delta([{"role": "tool", "content": "42"}]))
    assert "<tool>42|" in got


def test_separator_is_probed_from_the_template():
    # The fake template puts `~` after a closed assistant turn.
    assert Renderer(FakeTokenizer()).separator_ids == ids("~")


def test_extend_joins_prefix_emission_and_delta():
    r = Renderer(FakeTokenizer())
    prefix = ids("<user>hi|<assistant>done|")
    got = r.extend(prefix, [{"role": "user", "content": "next"}])
    assert r.decode(got) == "<user>hi|<assistant>done|~<user>next|<assistant>"


def test_extend_from_nothing_is_just_the_delta():
    r = Renderer(FakeTokenizer())
    assert r.extend([], [{"role": "user", "content": "hi"}]) == r.delta([{"role": "user", "content": "hi"}])


def test_extend_does_not_mutate_the_prefix():
    r = Renderer(FakeTokenizer())
    prefix = ids("<user>hi|")
    r.extend(prefix, [{"role": "user", "content": "x"}])
    assert prefix == ids("<user>hi|")


def test_decode_keeps_special_markers():
    r = Renderer(FakeTokenizer())
    assert r.decode(ids("<tool_call>")) == "<tool_call>"


def test_decode_of_nothing_is_empty_string():
    assert Renderer(FakeTokenizer()).decode([]) == ""


def test_template_kwargs_reach_the_template():
    seen = {}

    class Recording(FakeTokenizer):
        def apply_chat_template(self, conversation, **kw):
            seen.update(kw)
            return super().apply_chat_template(conversation, **kw)

    Renderer(Recording(), template_kwargs={"enable_thinking": False}).render([{"role": "user", "content": "x"}])
    assert seen["enable_thinking"] is False


def test_a_template_that_rewrites_its_prefix_is_an_error():
    class Rewriting(FakeTokenizer):
        def apply_chat_template(self, conversation, **kw):
            # Drops the fake prefix once anything is appended, so the render of
            # the longer conversation no longer starts with the shorter one.
            if len(conversation) > 2:
                conversation = conversation[2:]
            return super().apply_chat_template(conversation, **kw)

    with pytest.raises(RenderError, match="did not extend"):
        Renderer(Rewriting()).delta([{"role": "user", "content": "x"}])


class TestHooks:
    def test_seam_hook_replaces_the_default_join(self):
        # Stands in for a family whose emission already ends with the token the
        # delta begins with, so one has to go.
        r = Renderer(FakeTokenizer(), hooks=Hooks(seam=lambda prefix, delta: prefix[:-1] + delta))
        got = r.extend(ids("<user>hi|<assistant>done|"), [{"role": "user", "content": "n"}])
        assert r.decode(got) == "<user>hi|<assistant>done<user>n|<assistant>"

    def test_delta_hook_replaces_incremental_rendering(self):
        # For a tokenizer with no Jinja template to diff against.
        r = Renderer(FakeTokenizer(), hooks=Hooks(delta=lambda appended, tools: [1, 2, 3]))
        assert r.delta([{"role": "user", "content": "x"}]) == [1, 2, 3]

    def test_delta_hook_is_skipped_for_an_empty_append(self):
        r = Renderer(FakeTokenizer(), hooks=Hooks(delta=lambda appended, tools: [9]))
        assert r.delta([]) == []
