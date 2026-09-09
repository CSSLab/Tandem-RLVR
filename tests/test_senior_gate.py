"""A synthetic authorship mask zeroes junior positions in the gated response mask.

The function under test is `_apply_tandem_senior_gate` in
`verl/workers/utils/losses.py`, added by `third_party/verl-tandem.patch`. It is
the whole of the paper's senior only gradient claim: `ppo_loss` calls it on the
response mask before that mask reaches the policy loss, the entropy term and the
KL term, so a junior emitted position carries no gradient while still shaping the
group's reward.

The `denom` variable inside that function is computed from the UNGATED response
mask, and it feeds only the `actor/tandem_senior_token_frac` metric, not the
loss. That has been verified. The loss denominator comes from the mask this
function returns.

CPU only. Small tensors, no model, no distributed group.
"""

import types
import unittest

import torch

try:
    from verl.workers.utils.losses import _apply_tandem_senior_gate
    IMPORT_ERROR = None
except Exception as exc:  # verl absent or built without the tandem patch
    _apply_tandem_senior_gate = None
    IMPORT_ERROR = exc


# Two responses of length 6 in a batch padded to 8. Row 0 runs 6 tokens, row 1
# runs 4. Authorship alternates in runs, the way word handoff produces it.
RESPONSE_MASK = torch.tensor(
    [[1, 1, 1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)

# 1 = senior emitted, 0 = junior emitted. Column 6 of row 0 and column 4 of row 1
# are past the end of the response and must not survive the gate whatever they
# say.
TANDEM_MASK = torch.tensor(
    [[1, 1, 0, 0, 1, 0, 1, 0],
     [0, 1, 1, 0, 1, 1, 0, 0]], dtype=torch.int64)


def gate(jr_tkn_weight=0.0, response_mask=None, tandem_mask=TANDEM_MASK):
    config = types.SimpleNamespace(tandem_jr_tkn_weight=jr_tkn_weight)
    data = {} if tandem_mask is None else {"tandem_model_mask": tandem_mask}
    metrics = {}
    mask = RESPONSE_MASK if response_mask is None else response_mask
    return _apply_tandem_senior_gate(config, mask, data, metrics), metrics


@unittest.skipIf(_apply_tandem_senior_gate is None,
                 f"verl not importable: {IMPORT_ERROR}")
class SeniorGate(unittest.TestCase):

    def test_junior_positions_are_zeroed(self):
        gated, _ = gate()
        self.assertEqual(gated.tolist(),
                         [[1, 1, 0, 0, 1, 0, 0, 0],
                          [0, 1, 1, 0, 0, 0, 0, 0]])

    def test_padding_stays_zero_even_where_the_mask_claims_the_senior(self):
        gated, _ = gate()
        self.assertEqual(gated[0, 6].item(), False)
        self.assertEqual(gated[1, 4].item(), False)

    def test_the_gate_only_removes(self):
        gated, _ = gate()
        self.assertTrue(torch.all(gated <= RESPONSE_MASK),
                        "the gate must not add positions to the response mask")

    def test_no_mask_leaves_the_response_mask_alone(self):
        # This is the silent degradation path: build verl against stock vLLM and
        # no authorship mask reaches the batch, so training is plain GRPO.
        gated, metrics = gate(tandem_mask=None)
        self.assertIs(gated, RESPONSE_MASK)
        self.assertEqual(metrics, {})

    def test_an_all_senior_mask_is_a_no_op(self):
        gated, _ = gate(tandem_mask=torch.ones_like(TANDEM_MASK))
        self.assertEqual(gated.tolist(), RESPONSE_MASK.tolist())

    def test_junior_weight_scales_instead_of_gating(self):
        gated, _ = gate(jr_tkn_weight=0.25)
        self.assertEqual(gated.dtype, torch.float32)
        self.assertEqual(gated.tolist(),
                         [[1.0, 1.0, 0.25, 0.25, 1.0, 0.25, 0.0, 0.0],
                          [0.25, 1.0, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0]])

    def test_reported_senior_fraction_is_over_the_ungated_mask(self):
        _, metrics = gate()
        # 5 senior positions inside the 10 response positions.
        self.assertAlmostEqual(
            metrics["actor/tandem_senior_token_frac"].aggregate(), 0.5)

    def test_senior_fraction_is_reported_under_the_junior_weight_path_too(self):
        _, metrics = gate(jr_tkn_weight=0.25)
        self.assertAlmostEqual(
            metrics["actor/tandem_senior_token_frac"].aggregate(), 0.5)


if __name__ == "__main__":
    unittest.main()
