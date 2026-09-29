import pytest
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming
from pydantic import ValidationError

from tokin.apis import ChatRequest

# OpenAI's keys `tokin` declines, each because honouring it is beyond a token-level engine or would need work not yet done.
REFUSED = {
    "audio",
    "function_call",
    "functions",
    "logit_bias",
    "modalities",
    "moderation",
    "prediction",
    "reasoning_effort",
    "top_logprobs",
    "verbosity",
    "web_search_options",
}
ENGINE_EXTENSIONS = {"top_k", "min_p", "repetition_penalty"}

CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}


def request(**over):
    return ChatRequest.model_validate({"model": "m", "messages": [{"role": "user", "content": "hi"}], **over})


def test_every_openai_key_is_served_or_refused():
    # A bump of `openai` that adds a key fails here, so the new key gets a decision rather than a default.
    upstream = CompletionCreateParamsStreaming.__required_keys__ | CompletionCreateParamsStreaming.__optional_keys__
    served = {
        key
        for name, field in ChatRequest.model_fields.items()
        for key in (field.validation_alias.choices if field.validation_alias else [name])
    } - ENGINE_EXTENSIONS
    assert served <= upstream
    assert upstream - served == REFUSED


def test_undeclared_key_is_refused_by_name():
    with pytest.raises(ValidationError, match="verbosity"):
        request(verbosity="low")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("n", 2),
        ("stop", ["\n"]),
        ("tool_choice", "required"),
        ("logprobs", True),
        ("response_format", {"type": "json_object"}),
    ],
)
def test_what_cannot_be_honoured_is_refused_by_name(key, value):
    with pytest.raises(ValidationError, match=key):
        request(**{key: value})


def test_no_op_values_and_bookkeeping_pass():
    request(n=1, stop=[], tool_choice="auto", parallel_tool_calls=True, logprobs=False, user="u", store=False)


def test_messages_drop_unknown_keys():
    got = request(messages=[{"role": "user", "content": "hi", "provider_specific_fields": {}}])
    assert got.messages == [{"role": "user", "content": "hi"}]


def test_max_completion_tokens_is_max_tokens():
    assert request(max_completion_tokens=5).max_tokens == 5


def test_params_leave_out_what_the_harness_did_not_set():
    assert request(temperature=0.7).params(8, [1]) == {"max_tokens": 8, "stop_ids": [1], "temperature": 0.7}


def test_params_take_the_callers_max_tokens():
    assert request(max_tokens=100).params(8, [1])["max_tokens"] == 8


def test_lazy_fields_are_read_into_lists():
    parts = [{"type": "text", "text": "hi"}]
    got = request(messages=[{"role": "user", "content": parts}, {"role": "assistant", "tool_calls": [CALL]}])
    assert got.messages[0]["content"] == parts and got.messages[1]["tool_calls"] == [CALL]
