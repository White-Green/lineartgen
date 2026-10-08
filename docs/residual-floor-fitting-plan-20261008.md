# Residual-scale floor: matched fitting probe

Prepare two additional controlled fitting trajectories, based on the algebraic
scale audit while the original four-arm diagnostic is ongoing. The ongoing
four-arm study finishes unchanged. No additional broad development-set search
or production change is authorized by a fitting score alone.

Change only the conventional warm-started model's residual multiplier to
`sqrt(max(t,.05))` for t>0; use exact identity for t=0. The floor .05 is fixed to
the existing pixel-loss noise-weight floor, not selected by a quality sweep.
Keep the original raw guide and all126,828 parameters, clipped MSE and existing
loss coefficients. Compare `warm_floor` against `warm_hard` and
`warm_floor_interval` against `warm_interval` from the four-arm fitting probe.
The latter pair retains interval MSE, completing a 2×2 floor/gradient comparison;
the floor can interact with output saturation, so its effect is not assumed to
transfer between these losses. Both trajectories and thresholds are fixed before
any new fitting-trajectory update or quality evaluation; implementation tests use
small independent fixtures. An initial one-arm directory setup is kept
separately; no updates or evaluations ran from it.
Copy the frozen57 train pairs and57 local probe pairs byte-for-byte, and use the
same batch indices, Gaussian draws, masks, recursion1–6, sqrt(batch) LR, optimizer
reset, EMA, 1,200-update schedule and snapshots0/400/1200. No initialization from
the failed ink-guide candidates. The local probe was already used by the original
diagnostic and is not an unseen confirmation set.

Initial single-step outputs are identical wherever every pyramid strength is
zero or at least .05. Arbitrary partial masks and repeated sampling may introduce
smaller positive strengths, so blanket initial equivalence is not claimed.
Default floor0 preserves the old arithmetic and gradients exactly. The t=0
head gradient is zero with a positive floor. New ONNX Max/Greater/Where operators
are spatially pointwise; dynamic BGRA parity and protection require tests.

Before candidate updates: all tests, six-depth candidate GPU preflight and full
restore, and replay of both old warm six-update full model/optimizer/EMA states.
Freeze code/settings/input hashes. Do not start the new trajectory until the
original four-arm numerical comparison and native review are completed.

Use raw as the fitting primary, with EMA and partial masks diagnostic, exactly
as in the parent. This is a mechanism probe. It is promising for a broader study
only if its final mean blur F1 gains at least .03 against its matching loss
counterpart on both fixed train and local
probe sets, without clean F1 falling more than .01 or mean halo rising more than
.01 in any clean/tone/blur split, and native images support a real improvement.
Keep all failed individual fit flags. These advance-to-experiment rules do not
replace campaign adoption gates, independent training seeds, prospective draws,
or actual BGRA quality review. No extension past1,200 or alternative floor sweep.
