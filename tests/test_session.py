from tokin.rollout import Generation, Prompt
from tokin.session import Session


def test_starts_with_one_empty_rollout():
    s = Session()
    assert len(s.rollouts) == 1
    assert len(s.current) == 0


def test_turns_append_to_the_current_rollout():
    s = Session()
    s.current.append(Prompt([1, 2]))
    s.current.append(Generation([3], "stop", logprobs=[0.0]))
    assert len(s.rollouts) == 1
    assert s.current.token_ids == [1, 2, 3]


def test_fork_returns_a_fresh_current_and_keeps_the_previous_rollout():
    s = Session()
    s.current.append(Prompt([1, 2]))
    assert s.fork() is s.current
    s.current.append(Prompt([9]))
    assert [r.token_ids for r in s.rollouts] == [[1, 2], [9]]


def test_each_rollout_may_open_with_a_prompt_after_forking():
    # A fork resets the "first segment must be a prompt" state, so the new rollout
    # is a valid trajectory on its own.
    s = Session()
    s.current.append(Prompt([1]))
    s.current.append(Generation([2], "stop", logprobs=[0.0]))
    s.fork().append(Prompt([3]))
    s.current.append(Generation([4], "stop", logprobs=[0.0]))
    assert [len(r.segments) for r in s.rollouts] == [2, 2]


def test_repr_summarises_without_dumping_tokens():
    s = Session()
    s.current.append(Prompt([1, 2, 3]))
    s.fork()
    s.current.append(Prompt([4]))
    assert repr(s) == f"Session(id={s.id!r}, rollouts=2, tokens=4)"


def test_sessions_do_not_share_their_default_rollout():
    a, b = Session(), Session()
    a.current.append(Prompt([1]))
    assert len(b.current) == 0


def test_id_is_fresh_unless_given():
    assert Session().id != Session().id and len(Session().id) == 32
    assert Session(id="episode-7").id == "episode-7"
