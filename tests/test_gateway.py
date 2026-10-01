import pytest
from conftest import END

from tokin.backends import GenerationBackend, GenerationError
from tokin.gateway import (
    DuplicateSessionError,
    Gateway,
    GatewayError,
    GenerationAbortedError,
    HistoryConflictError,
    PromptTooLongError,
    SessionNotFoundError,
)
from tokin.rollout import FinishReason, Generation

USER = {"role": "user", "content": "hi"}
REPLY = {"role": "assistant", "content": "ok"}
AGAIN = {"role": "user", "content": "again"}


class Engine(GenerationBackend):
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    async def context_length(self):
        return None

    async def generate(self, token_ids, params):
        self.calls.append((token_ids, params))
        return self.replies.pop(0)


def said(text, finish=FinishReason.STOP):
    return Generation([ord(c) for c in text] + [ord(END)] * (finish is FinishReason.STOP), finish)


@pytest.fixture
def gateway(chatml):
    def make(*replies, context_length=1000):
        return Gateway(chatml, Engine(*replies), context_length=context_length)

    return make


def test_create_mints_an_id_unless_given(gateway):
    g = gateway()
    assert g.create().id != g.create().id
    assert g.create("episode-7").id == "episode-7"
    assert len(g.sessions) == 3


def test_get_returns_the_same_session(gateway):
    g = gateway()
    assert g.get(g.create("a").id) is g.sessions["a"]


def test_delete_removes_and_returns(gateway):
    g = gateway()
    s = g.create("a")
    assert g.delete("a") is s
    assert not g.sessions


@pytest.mark.parametrize("call", [Gateway.get, Gateway.delete])
def test_unknown_session_is_404(gateway, call):
    with pytest.raises(SessionNotFoundError, match="'nope'") as e:
        call(gateway(), "nope")
    assert e.value.status_code == 404


def test_taken_id_is_409(gateway):
    g = gateway()
    g.create("a")
    with pytest.raises(DuplicateSessionError, match="'a'") as e:
        g.create("a")
    assert e.value.status_code == 409 and isinstance(e.value, GatewayError)


class TestChat:
    async def test_first_turn_renders_in_full(self, gateway, chatml):
        g = gateway(said("ok"))
        message, generation, prompt_tokens = await g.chat([USER], session_id=g.create().id)
        assert g.backend.calls[0][0] == chatml.encode(chatml.apply([USER])) and prompt_tokens == len(
            g.backend.calls[0][0]
        )
        assert message == {"role": "assistant", "content": "ok"} and generation.finish_reason is FinishReason.STOP

    async def test_later_turn_extends_the_rollout_by_the_increment(self, gateway, chatml):
        g = gateway(said("ok"), said("ok"))
        s = g.create()
        await g.chat([USER], session_id=s.id)
        served = s.current.token_ids
        await g.chat([USER, REPLY, AGAIN], session_id=s.id)
        assert g.backend.calls[1][0] == served + chatml.encode(chatml.apply_increment([AGAIN]))

    async def test_rewritten_history_is_a_conflict(self, gateway):
        g = gateway(said("ok"))
        s = g.create()
        await g.chat([USER], session_id=s.id)
        with pytest.raises(HistoryConflictError, match="extend"):
            await g.chat([{"role": "user", "content": "changed"}, REPLY, AGAIN], session_id=s.id)

    async def test_assistant_tokin_did_not_write_is_a_conflict(self, gateway):
        g = gateway(said("ok"))
        s = g.create()
        await g.chat([USER], session_id=s.id)
        with pytest.raises(HistoryConflictError, match="did not generate"):
            await g.chat([USER, REPLY, AGAIN, {"role": "assistant", "content": "Sure, "}], session_id=s.id)

    async def test_cut_short_generation_cannot_be_extended(self, gateway):
        g = gateway(said("o", FinishReason.LENGTH))
        s = g.create()
        await g.chat([USER], session_id=s.id)
        with pytest.raises(HistoryConflictError, match="stop token"):
            await g.chat([USER, {"role": "assistant", "content": "o"}, AGAIN], session_id=s.id)

    async def test_prompt_past_the_context_is_refused(self, gateway):
        with pytest.raises(PromptTooLongError):
            await gateway(context_length=5).chat([USER])

    async def test_max_tokens_is_capped_to_the_room_left(self, gateway, chatml):
        g = gateway(said("ok"), context_length=len(chatml.encode(chatml.apply([USER]))) + 3)
        await g.chat([USER], params={"max_tokens": 100})
        assert g.backend.calls[0][1]["max_tokens"] == 3

    async def test_stop_off_a_stop_id_is_an_engine_error(self, gateway):
        with pytest.raises(GenerationError, match="stop id"):
            await gateway(Generation([ord("o")], FinishReason.STOP)).chat([USER])

    async def test_engine_abort_is_503_and_records_nothing(self, gateway):
        g = gateway(said("o", FinishReason.ABORT))
        s = g.create()
        with pytest.raises(GenerationAbortedError) as e:
            await g.chat([USER], session_id=s.id)
        assert e.value.status_code == 503 and not s.current.segments and not s.current.messages

    async def test_without_a_session_nothing_is_kept(self, gateway):
        g = gateway(said("ok"))
        await g.chat([USER])
        assert not g.sessions
