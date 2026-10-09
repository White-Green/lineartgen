# Startup replay exception before update7

The frozen experiment stopped at its required exact six-update replay check.
Both preflight and first actual run included the same step0 evaluation. Their
model parameters differ by at most1.85519457e-6, optimizer state3.71411443e-6 and
EMA2.98023224e-8. A separate identical-startup repeat also differs: versus the
first actual run, maxima are4.61004674e-6 /6.71390444e-5 /2.98023224e-8. The repeat
verified that initial snapshot/evaluation does not mutate the model, optimizer
or EMA. This does not identify the exact responsible GPU primitive.

Before the output-sensitivity check, fixed its threshold at1e-4 in model output
units (less than1/78 of one8-bit step on[-1,1]). On eight fixed geometry sources,
the first actual6 versus preflight6 maximum prediction difference is4.17232513e-7.
All inputs, seeds, configuration and source bindings remain identical. The
historical control's independent six-update replay is still bitwise exact.

Amend only the requirement for bitwise equality between separate training runs:
record this known exception and continue the ORIGINAL FIRST actual6 checkpoint,
archived with file hashes under startup-checkpoint/6. Do not substitute the later
repeat, change losses, thresholds, data, endpoint or choose an intermediate model.
Keep all predeclared quality gates, full-state restore checks and raw/EMA export
agreement exact. The numerical source remains f81709e; continuation uses its
existing resume path from6, with this explicit addendum and a bound coordinator.

This is diagnostic evidence, not a reproducibility or deployment claim. Keep
all_state_exact=false in replay-audit.json. Any successful quality result still
needs independent-seed training and the unopened confirmation protocol.

[OpenXLA's determinism documentation](https://openxla.org/xla/determinism)
describes both compilation choices and execution-order effects as possible
sources of GPU floating-point variation, and deterministic modes may alter
kernel selection or performance. We have not proven which mechanism caused this
run's difference, and have not silently changed backend flags for one arm.

Evidence: startup-difference.json, replay-after-evaluation-diagnostic.json,
startup-sensitivity-plan.json, startup-sensitivity.json and startup-checkpoint/
binding.json in runs/signal-transport-20261009. The failed startup log remains.
