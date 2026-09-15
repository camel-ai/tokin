import pytest

from tokin.rollout import Rollout, Turn


def test_turn_rejects_misaligned_logprobs():
    with pytest.raises(ValueError, match="2 entries but token_ids has 3"):
        Turn(token_ids=[1, 2, 3], generated=True, logprobs=[0.0, 0.0])


def test_generated_turn_may_omit_logprobs():
    assert Turn(token_ids=[1, 2], generated=True).logprobs is None


def test_token_ids_concatenate_in_turn_order():
    r = Rollout()
    r.add_prompt([1])
    for i in range(3):
        r.add_response([100 + i], logprobs=[0.0])
        r.add_prompt([200 + i])
    assert r.token_ids == [1, 100, 200, 101, 201, 102, 202]


def test_generated_flag_follows_the_call_used():
    r = Rollout()
    r.add_prompt([1])
    r.add_response([2], logprobs=[0.0])
    assert [t.generated for t in r.turns] == [False, True]


def test_empty_turns_are_dropped():
    r = Rollout()
    r.add_prompt([1])
    r.add_prompt([])
    r.add_response([], logprobs=[])
    assert len(r.turns) == 1


def test_inputs_are_copied_not_aliased():
    ids, logprobs = [1, 2], [-0.1, -0.2]
    r = Rollout()
    r.add_prompt(ids)
    r.add_response(ids, logprobs=logprobs)
    ids.append(99)
    logprobs.append(99.0)
    assert r.token_ids == [1, 2, 1, 2]
    assert r.turns[1].logprobs == [-0.1, -0.2]


def test_token_ids_returns_a_fresh_list():
    # Callers concatenate onto it to build the next prompt.
    r = Rollout()
    r.add_prompt([1])
    r.token_ids.append(2)
    assert r.token_ids == [1]


def test_repr_summarises_without_dumping_tokens():
    r = Rollout()
    r.add_prompt([1, 2, 3])
    r.add_response([4, 5], logprobs=[0.0, 0.0])
    assert repr(r) == "Rollout(turns=2, tokens=5, generated=2)"


def test_add_response_cannot_open_a_rollout():
    # A loss mask starting at 1 would train on tokens nothing conditioned on.
    with pytest.raises(RuntimeError, match="cannot open with a generated turn"):
        Rollout().add_response([1], logprobs=[0.0])


def test_empty_response_before_any_prompt_is_not_an_error():
    r = Rollout()
    r.add_response([])
    assert r.turns == []
