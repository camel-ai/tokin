from tokin.session import Session


def test_starts_with_one_empty_rollout():
    s = Session()
    assert len(s.rollouts) == 1
    assert len(s.current) == 0


def test_turns_append_to_the_current_rollout():
    s = Session()
    s.current.add_prompt([1, 2])
    s.current.add_response([3], logprobs=[0.0])
    assert len(s.rollouts) == 1
    assert s.current.token_ids == [1, 2, 3]


def test_fork_leaves_the_previous_rollout_intact():
    s = Session()
    s.current.add_prompt([1, 2])
    s.fork()
    s.current.add_prompt([9])
    assert [r.token_ids for r in s.rollouts] == [[1, 2], [9]]


def test_fork_returns_the_new_current():
    s = Session()
    assert s.fork() is s.current


def test_length_spans_every_rollout():
    s = Session()
    s.current.add_prompt([1, 2])
    s.fork()
    s.current.add_prompt([3, 4, 5])
    assert len(s) == 5


def test_each_rollout_may_open_with_a_prompt_after_forking():
    # A fork resets the "first turn must be a prompt" state, so the new rollout
    # is a valid trajectory on its own.
    s = Session()
    s.current.add_prompt([1])
    s.current.add_response([2], logprobs=[0.0])
    s.fork().add_prompt([3])
    s.current.add_response([4], logprobs=[0.0])
    assert [len(r.turns) for r in s.rollouts] == [2, 2]


def test_repr_summarises_without_dumping_tokens():
    s = Session()
    s.current.add_prompt([1, 2, 3])
    s.fork()
    s.current.add_prompt([4])
    assert repr(s) == "Session(rollouts=2, tokens=4)"


def test_sessions_do_not_share_their_default_rollout():
    a, b = Session(), Session()
    a.current.add_prompt([1])
    assert len(b.current) == 0
