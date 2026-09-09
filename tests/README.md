# Tests

Two groups. The files in this directory run on CPU in under a second. The `gpu/` group loads

## CPU

```bash
python -m unittest discover -s tests
```

| file | what it proves |
|---|---|
| `test_word_state_machine.py` | the word handoff redraws the active model only at a boundary token or when `max_gap_tokens` is reached, and the recovery path reproduces the live one |
| `test_senior_gate.py` | a junior emitted position carries no gradient: the authorship mask zeroes it in the response mask `ppo_loss` uses |

Both import the function under test out of the installed fork rather than restating the rule
in the test, so they follow the fork instead of drifting from it. Neither needs a model, a
GPU or a distributed group: the word schedule is plain python over token ids, and the gate
is a tensor operation on a 2x8 mask.

Both skip themselves when the symbol they test is not importable, and a skip is not a pass.
Read the skip reason: a missing package and an unpatched package look different there, and
the second is the one to worry about, since under stock vLLM the authorship mask never
arrives and training silently degrades to plain GRPO.

## GPU

