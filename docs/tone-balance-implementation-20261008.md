# Tone-weight implementation and validation

`loss.tone_balance_power` now selects the strength of inverse-frequency teacher
tone weighting. Its default1 preserves the prior expression exactly;0 gives unit
weights exactly; intermediate values exponentiate the old weights and normalize
their spatial mean per image. The existing `balance_tones=false` option remains
equivalent to unit weights. Model inference and default training are unchanged.

All284 CPU tests pass in553.89 seconds, with three inherited deprecation warnings.
Five new tests cover exact legacy weights, unit-weight loss/gradient equivalence,
finite normalization on blank/full-ink images, the reduced rare-ink ratio,
stationarity of the actual tempered pixel objective, and parameter validation.
The focused run passed in4.35 seconds.100 source/test files have unchanged hashes
after the full run. Preparation, comparison and native rendering helpers pass
Ruff and compilation checks.

The [fixed local plan](tone-balance-fitting-plan-20261008.md) is prepared under
`runs/tone-balance-fitting-20261008/`. Numerical source is frozen at
2aa86743d19f477609a56e9d949dd9684168c5b4,233 files, five settings for the two
candidates and four for a separate historical replay. Preparation verified exact
data copies and that only the tone exponent differs from the historical width16
configuration. Both models still have130,605 parameters and restart from the
original100k EMA. No locally fitted weights initialize them.

The independent conditional-path evaluation completed and was rejected. Both
tone candidates then passed six-depth GPU updates with exact full-state restore.
The historical width16 six-update replay and all36 initial forward comparisons
are exact. The first actual update restored exactly before the fixed trajectories
continued. The runner evaluates0/400/1,200 and stops after comparison and rendering
twelve sheets for actual inspection. The separate partial-summary fix has two
focused tests passing here in.93 seconds; frozen training/inference sources and
conditions are unchanged. Learned quality remains pending. No replacement is adopted.

The two experiments isolate different changes. A failure of tone tempering on
the ordinary rough path would not rule out an interaction with bridge training.
Teacher-dependent pixel weights change the conditional clean-image average, so
the current balanced bridge objective is not the plain conditional-flow-matching
objective. No combined path/loss experiment is implemented or scheduled here.
