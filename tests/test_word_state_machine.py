"""The word handoff schedule redraws the active model only at a boundary token or max_gap.

The rule under test lives in `vllm/v1/sample/tandem_sampler.py`, added by
`third_party/vllm-tandem.patch`. Each request carries `[author_next,
since_last_redraw]`. Every emitted token increments the counter. The active model
is redrawn when that token is in `boundary_token_ids`, or when the counter
reaches `max_gap_tokens`, and the counter resets on a redraw. Nothing else
redraws: not a step boundary, not a batch rebuild, not another request's tokens.

Two code paths implement the rule and this file tests both against each other.
`_word_post_update` advances the state with the token just sampled, which is the
path a running request takes. `_word_replay` rebuilds the state by scanning a
token prefix, which is the recovery path for a request whose state was lost, and
it skips negative ids: under async scheduling a prefix carries placeholders for
tokens that have not synced yet.

CPU only. The schedule is plain python over token ids, so no model and no GPU are
needed. `TandemSampler.__init__` is bypassed because it builds a vLLM `Sampler`
the schedule never touches.
"""

import unittest

import torch

try:
    from vllm.v1.sample.tandem_sampler import TandemSampler
    IMPORT_ERROR = None
except Exception as exc:  # vLLM absent or built without the tandem patch
    TandemSampler = None
    IMPORT_ERROR = exc


class ScriptedCoin:
    """Replaces `_draw_active` so that a redraw is observable.

    The real draw is a random number against `prob_primary`. Two consecutive
    draws that happen to agree are indistinguishable from no draw at all, so a
    test of when redraws happen has to control the sequence and count the calls.
    """

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self, _metadata, _index, _prob):
        value = self.values[self.calls % len(self.values)]
        self.calls += 1
        return value


class FakeMetadata:
    """The two attributes of SamplingMetadata the word schedule reads."""

    def __init__(self):
        self.generators = {}
        self.output_token_ids = []


def make_sampler(boundary_ids=(), max_gap=32, coin_values=(True, False)):
    sampler = TandemSampler.__new__(TandemSampler)
    sampler.strategy = "word"
    sampler.prob_primary = 0.5
    sampler.chunk_size = 1
    sampler.max_gap_tokens = max_gap
    sampler._step = 0
    sampler._word_state = {}
    sampler.boundary_token_ids = frozenset(boundary_ids)
    coin = ScriptedCoin(coin_values)
    sampler._draw_active = coin
    return sampler, coin


def run_incremental(sampler, tokens, key="req0"):
    """Drive the live path: seed the state, then emit `tokens` one per step."""
    metadata = FakeMetadata()
    sampler._word_state[key] = sampler._word_replay([], metadata, 0)
    for token in tokens:
        sampler._word_post_update([key], [token], [True], metadata)
    return sampler._word_state[key]


@unittest.skipIf(TandemSampler is None, f"vllm not importable: {IMPORT_ERROR}")
class WordSchedule(unittest.TestCase):

    def test_no_redraw_without_a_boundary_or_gap(self):
        sampler, coin = make_sampler(boundary_ids={100}, max_gap=32,
                                     coin_values=[True, False])
        state = run_incremental(sampler, [1, 2, 3, 4, 5])
        self.assertEqual(coin.calls, 1, "only the initial draw should have run")
        self.assertIs(state[0], True)
        self.assertEqual(state[1], 5, "counter advances on every token")

    def test_redraw_at_a_boundary_token(self):
        sampler, coin = make_sampler(boundary_ids={100}, max_gap=32,
                                     coin_values=[True, False])
        state = run_incremental(sampler, [1, 2, 100, 3])
        self.assertEqual(coin.calls, 2)
        self.assertIs(state[0], False, "the boundary token handed over")
        self.assertEqual(state[1], 1, "counter restarted at the boundary")

    def test_redraw_when_max_gap_is_reached(self):
        sampler, coin = make_sampler(boundary_ids=set(), max_gap=4,
                                     coin_values=[True, False])
        state = run_incremental(sampler, [1, 2, 3, 4, 5])
        self.assertEqual(coin.calls, 2, "redraw on the 4th token, not the 5th")
        self.assertIs(state[0], False)
        self.assertEqual(state[1], 1)

    def test_gap_counter_restarts_after_a_boundary(self):
        # A boundary at token 1 must reset the gap counter, so the next forced
        # redraw is max_gap tokens later rather than at a fixed step number.
        sampler, coin = make_sampler(boundary_ids={7}, max_gap=3,
                                     coin_values=[True, False, True])
        state = run_incremental(sampler, [7, 1, 1, 1])
        self.assertEqual(coin.calls, 3, "one initial, one boundary, one gap")
        self.assertIs(state[0], True)
        self.assertEqual(state[1], 0)

    def test_no_state_means_no_update(self):
        sampler, coin = make_sampler(boundary_ids={100})
        metadata = FakeMetadata()
        sampler._word_post_update(["unknown"], [100], [True], metadata)
        self.assertEqual(coin.calls, 0)
        self.assertEqual(sampler._word_state, {})

    def test_replay_skips_async_placeholders(self):
        # -1 marks a token whose value has not synced back yet. Counting it
        # would move the gap counter, and it can never match a boundary id, so
        # skipping is the only reading that survives async scheduling.
        sampler, coin = make_sampler(boundary_ids=set(), max_gap=3,
                                     coin_values=[True, False])
        state = sampler._word_replay([1, -1, -1, 2], FakeMetadata(), 0)
        self.assertEqual(coin.calls, 1, "placeholders must not force a redraw")
        self.assertEqual(state[1], 2, "only the two real tokens counted")

    def test_replay_of_only_placeholders_is_a_fresh_state(self):
        sampler, coin = make_sampler(boundary_ids=set(), max_gap=3,
                                     coin_values=[True, False])
        state = sampler._word_replay([-1] * 10, FakeMetadata(), 0)
        self.assertEqual(coin.calls, 1)
        self.assertEqual(state[1], 0)

    def test_replay_reproduces_the_incremental_state(self):
        # A request preempted and resumed must land where it would have been.
        tokens = [1, 2, 7, 3, 4, 5, 6, 7, 8, 9]
        live, live_coin = make_sampler(boundary_ids={7}, max_gap=4,
                                       coin_values=[True, False, True, False])
        replayed, replay_coin = make_sampler(boundary_ids={7}, max_gap=4,
                                             coin_values=[True, False, True, False])
        self.assertEqual(run_incremental(live, tokens),
                         replayed._word_replay(tokens, FakeMetadata(), 0))
        self.assertEqual(live_coin.calls, replay_coin.calls)


@unittest.skipIf(TandemSampler is None, f"vllm not importable: {IMPORT_ERROR}")
class WordSelect(unittest.TestCase):
    """`_word_select` reports the current author and owns the state lifetime."""

    def test_reports_the_stored_author_per_request(self):
        sampler, _ = make_sampler(boundary_ids={100}, coin_values=[True, False])
        chosen = sampler._word_select(2, torch.device("cpu"), FakeMetadata(),
                                      req_ids=["a", "b"],
                                      output_token_ids=[[], []])
        self.assertEqual(chosen.dtype, torch.bool)
        self.assertEqual(chosen.tolist(),
                         [sampler._word_state["a"][0],
                          sampler._word_state["b"][0]])

    def test_state_of_a_finished_request_is_dropped(self):
        # State is keyed by request id, not by the row's list object: the
        # persistent batch reuses list objects across steps, so a stale entry
        # would be handed to whatever request inherits the row.
        sampler, _ = make_sampler(boundary_ids={100}, coin_values=[True])
        sampler._word_select(2, torch.device("cpu"), FakeMetadata(),
                             req_ids=["a", "b"], output_token_ids=[[], []])
        self.assertEqual(set(sampler._word_state), {"a", "b"})
        sampler._word_select(1, torch.device("cpu"), FakeMetadata(),
                             req_ids=["a"], output_token_ids=[[]])
        self.assertEqual(set(sampler._word_state), {"a"})


if __name__ == "__main__":
    unittest.main()
