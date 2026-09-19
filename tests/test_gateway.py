import pytest

from tokin.gateway import DuplicateSessionError, Gateway, GatewayError, SessionNotFoundError


def test_create_mints_an_id_unless_given():
    g = Gateway()
    assert g.create().id != g.create().id
    assert g.create("episode-7").id == "episode-7"
    assert len(g.sessions) == 3


def test_get_returns_the_same_session():
    g = Gateway()
    assert g.get(g.create("a").id) is g.sessions["a"]


def test_delete_removes_and_returns():
    g = Gateway()
    s = g.create("a")
    assert g.delete("a") is s
    assert not g.sessions


@pytest.mark.parametrize("call", [Gateway.get, Gateway.delete])
def test_unknown_session_is_404(call):
    with pytest.raises(SessionNotFoundError, match="'nope'") as e:
        call(Gateway(), "nope")
    assert e.value.status_code == 404


def test_taken_id_is_409():
    g = Gateway()
    g.create("a")
    with pytest.raises(DuplicateSessionError, match="'a'") as e:
        g.create("a")
    assert e.value.status_code == 409 and isinstance(e.value, GatewayError)
