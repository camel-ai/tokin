from tokin.rollout import Generation, Prompt
from tokin.session import Session


def test_starts_with_one_empty_rollout():
    s = Session()
    assert len(s.rollouts) == 1
    assert len(s.current) == 0


def test_turns_append_to_the_current_rollout():
    s = Session()
    s.current.append(Prompt([1, 2]), [])
    s.current.append(Generation([3], "stop", logprobs=[0.0]), [])
    assert len(s.rollouts) == 1
    assert s.current.token_ids == [1, 2, 3]


def test_repr_summarises_without_dumping_tokens():
    s = Session()
    s.current.append(Prompt([1, 2, 3]), [])
    assert repr(s) == f"Session(id={s.id!r}, rollouts=1, tokens=3)"


def test_sessions_do_not_share_their_default_rollout():
    a, b = Session(), Session()
    a.current.append(Prompt([1]), [])
    assert len(b.current) == 0


def test_sessions_do_not_share_a_lock():
    assert Session().lock is not Session().lock


def test_id_is_fresh_unless_given():
    assert Session().id != Session().id and len(Session().id) == 32
    assert Session(id="episode-7").id == "episode-7"
