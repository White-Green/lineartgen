# AD: near-line paper-band fitting — completed

Adding paper_band_weight=.1 to AA improves the fixed fitting diagnostic from
3/22 failed checks to1/22. All four trained cases now pass halo<=.003; cuts and
measured bridges remain zero. The curve still fails F1>=.98. **Not adopted.**

Only this coefficient differs from AA. Both start from the same R raw4k weights,
with fresh optimizer/EMA,130,605 parameters, the same four fixed pairs, and1,200
updates cycling depths1–6 at constant sqrt(batch)-scaled LR1e-4. AA controls are
reused, not retrained. Final raw1200 was specified before training.

| Trained case | AA F1 → AD F1 | AA halo → AD halo | AD ink deficit |
| --- | --- | --- | ---: |
| Curve999 | .960748 → .958879 | .001232 → .001313 | .006116 |
| Corner1095 | .992200 → .993730 | .003092 → .001160 | .001999 |
| Parallel1287 | .997799 → .998532 | .005962 → .001010 | .001445 |
| Gap1383 | .994048 → .991098 | .000064 → .000121 | .009579 |

All eight prespecified trained-case regression checks pass: F1 decline<=.005,
ink-deficit increase<=.01 versus AA. Failure counts at0/100/400/1200 are
20/8/2/1 for AD versus20/12/4/3 for AA. This supports the paper-gradient mechanism,
not generalization or complete lineart quality.

Unfitted probes expose limitations. Straight807 deficit rises .044149→.091230.
Junction1191 F1 falls .775674→.740445,deficit .169765→.211367,cuts .133333→.333333.
Thin903 improves F1 .980597→.986627 and deficit .010711→.005679. Thick taper1479
still looks gray and uneven: deficit .302637→.254504,cuts .333333→0,halo
.009481→.013237. Both final native sheets were viewed and hash-bound. Fitted
weights must not be adopted or used to initialize the broad follow-up.

The teacher-paper mask and physical preclip gradient were checked on the four
actual initial examples. It selects paper alongside open curves and corners as
well as close pairs/gaps; direct teacher-ink gradient above alpha.02 is zero and
false black saturation receives an inward gradient. Shared weights still affect
ink, as the probes demonstrate.30 tests pass, including unchanged default loss/
gradients, actual update/full-state/next-update restore and generic preclip routes.
Initial outputs match at all6 depths. All8 AA predictions and metrics reproduce
exactly at each milestone; actual6/100/400/1200 states and6 raw/EMA exports match.

Numerical commit `d29885deeaa4fdc6831b11f26a4e55d62a957321`.
Evidence: `runs/paper-band-fit-20261010/final-audit.json`.
No independent/sealed evaluation, production replacement or plugin changes.
The next diagnostic will add this one term to AB's broad protocol, retain its
original teacher mixture and initialize from E. It will explicitly inspect the
unfitted junction/thick-line regressions under the unchanged229 broad checks.
