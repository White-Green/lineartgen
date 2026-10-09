# Opacity-retention preparation and validation

Implementation and numerical plan are frozen at `c2ceb6b04241f100a98022dd4001ba4be93a0d56` in `experiment/opacity-retention-fitting-20261009`. The two candidate configurations differ from the completed square-root-weight configuration only in `loss.opacity_deficit_weight` (.1/.3). Their data match exactly. Inference, architecture, initialization and sampler have no changes.

The full CPU suite passed: **292 tests, 3 inherited deprecation warnings, 481.63 seconds**. The 103 source/test files bound at test start are unchanged. The focused opacity/tone/summary run also passed all 13 tests. Ruff and script compilation passed. Hashes and logs are in `runs/opacity-retention-validation-20261009/result.json` and its neighboring files.

Both candidate and historical-replay roots are frozen: 240 tracked numerical/tool/test/package files, five candidate settings and four replay settings. Source archives, original model files, data files, the declared plan, and both existing references are bound by hashes. The new term is disabled by default. The meaningful legacy numerical check remains the queued exact six-update GPU replay, in addition to the completed CPU regressions.

The queued runner is `runs/opacity-retention-fitting-20261009/fitting_runner.py`, with its separate hash in `execution-plan.json`. It requires the validated CPU result and waits for the already running plain-path study to complete numerical evaluation before touching the GPU. It then performs candidate six-depth updates/full restore, exact historical replay, 36 exact initial forward comparisons, actual update1/full restore, and the fixed 0/400/1200 evaluations and final native rendering. Actual GPU preflight and candidate training have **not** yet started as of this preparation record.

Completion of training is not adoption: the runner stops with native review pending. The primary gate stays the balanced-reference raw one-step gate in the plan; matched square-root differences are separately reported. No model is copied into the Krita plugin by this workflow.
