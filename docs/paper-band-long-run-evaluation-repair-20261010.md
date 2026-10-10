# AP evaluator repair

The first run stopped after six successful updates (4006) during local4000
summary: the inherited evaluator hardcoded a control called pixel, whereas AP's
local comparison is AE4000, named band4000. All912 broad outputs and128 primary
outputs were already produced. This is an evaluation wiring error, not a quality
result. The original frozen worktree and artifacts remain unchanged.

This isolated worktree adds local_control_method with backward-compatible pixel
default, and sets it to band4000 in AP. Unit checks replay all old local summary
rules both with the default and renamed control. The main229 E/R adoption rules,
model/loss/configuration/data and16000 endpoint are unchanged. No output-based
selection or learning-rate changes are introduced. The AP declaration remains
the same document fixed before AO results.

Replay the six startup updates from the same actual AE4000 state and require
bitwise equality of model, optimizer, EMA, all six sampling records and initial
predictions with the interrupted run before further training. Bind that evidence.
This replay is not independent training evidence. Continue the same planned
12000-update extension only after that audit and repaired local summary pass.
