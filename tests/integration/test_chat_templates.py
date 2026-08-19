"""Does incremental rendering hold on the templates real models ship?

Downloads tokenizers, so it is marked and skipped by default. Run with:
    uv run pytest tests/integration -m tokenizer

These check the two properties the gateway depends on: that appending turns to an
inert prefix yields a delta, and that concatenating ids is the same as encoding
the concatenated text. Both are properties of upstream templates, so a failure
here is news about a model, not necessarily a bug in tokin.
"""

import pytest

from tokin.render import Renderer

pytestmark = pytest.mark.tokenizer

FAMILIES = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-4B",
    "zai-org/GLM-4.5",
    "zai-org/GLM-5.2",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3.1",
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2.6",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "look up",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]
TOOL_CALLS = [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": {"q": "x"}}}]

APPENDS = [
    ("user", [{"role": "user", "content": "And 3+3?"}], None),
    ("user with tools", [{"role": "user", "content": "And 3+3?"}], TOOLS),
    ("tool result", [{"role": "tool", "tool_call_id": "call_1", "content": "42"}], TOOLS),
]


@pytest.fixture(scope="module")
def renderers():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    out = {}
    for name in FAMILIES:
        try:
            tok = transformers.AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        except Exception as exc:  # gated, offline, or missing remote code
            pytest.skip(f"{name} unavailable: {type(exc).__name__}")
        out[name] = Renderer(tok)
    return out


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize(("case", "appended", "tools"), APPENDS, ids=[a[0] for a in APPENDS])
def test_appending_yields_a_delta(renderers, model, case, appended, tools):
    delta = renderers[model].delta(appended, tools)
    assert delta, f"{model} produced no delta for {case}"


@pytest.mark.parametrize("model", FAMILIES)
def test_delta_contains_the_appended_content(renderers, model):
    r = renderers[model]
    text = r.decode(r.delta([{"role": "user", "content": "UNIQUEMARKER"}]))
    assert "UNIQUEMARKER" in text


@pytest.mark.parametrize("model", FAMILIES)
def test_concatenating_ids_matches_encoding_the_joined_text(renderers, model):
    # The gateway concatenates ids across turns, so this equality is what keeps a
    # spliced prompt identical to one rendered in a single pass.
    r = renderers[model]
    first = r.render([{"role": "user", "content": "What is 2+2?"}])
    emission = r.encode("It is 4.")
    delta = r.delta([{"role": "user", "content": "And 3+3?"}])
    joined = r.encode(r.decode(first) + r.decode(emission) + r.decode(delta))
    assert first + emission + delta == joined


@pytest.mark.parametrize("model", FAMILIES)
def test_separator_probe_returns_something_decodable(renderers, model):
    # An empty separator is a valid answer: some templates put nothing between a
    # closed assistant turn and the next message.
    sep = renderers[model].separator_ids
    assert isinstance(sep, list)
    renderers[model].decode(sep)
