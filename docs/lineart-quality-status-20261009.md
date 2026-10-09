# Lineart restoration experiment status, 2026-10-09

No replacement has been adopted. The deployed original 100k model remains the baseline. Synthetic rough inputs remain the only corruption source. The required combination—less gray fringe with preserved thin lines, opacity, width, continuity and separation—has not yet passed the complete evaluation.

Recent results explain the two experiments now in progress:

| Experiment | What it established | Decision |
| --- | --- | --- |
| Full-resolution width16 detail head | Improved local fitting, roughly +.10 blur F1 over the old local guided model; broader results still have gray marks, missing fine strokes, or wrong joins | Broader candidate rejected after native review |
| Progress-conditioned rough/bridge paths with full tone balancing | The learned five-step bridge still forms dark branching structures; blur F1 is .2115/.2488 on train/probe versus .5304/.5804 for the old one-step local reference | Both paths insufficient; bridge advancement gate rejected |
| Tempered tone weights | Square-root weights reduce low-strength blur halos by 47.49%/49.43%, but all 24 strata exceed allowed opacity loss; unit weights fade lines further | Both rejected; all 12 final native sheets inspected |
| Unit-weight rough/bridge paths | Much less false branching, but bridge still loses thin strokes; fails10/18 comparisons after1,200 updates and all10 native sheets reviewed | Declared bridge candidate rejected |
| Square-root weights plus linear opacity deficit | Directly targets the opacity lost by tempering, at fixed coefficients .1 and .3 | GPU preflight/replay/initial audits and actual update1 restore passed; running in `jax-opacity-retention-20261009` |

The opacity term measures missing opacity only at editable teacher-line pixels. It has no added force at an already correct gray edge, a protected pixel, or the background. This is a hypothesis about balancing losses, not a guarantee of correct geometry or a proof of the cause of halos. It can still cause excessive darkening under uncertainty; fixed metric safeguards and actual output inspection decide the result.

The queued study starts from the original 100k EMA, not a selected fitted endpoint, with the same local 57/57 pairs and random streams. Architecture, recursive depths1–6, stage weighting0, downsample-before-noise, sqrt(batch) learning-rate scaling, and the fixed 1,200-update schedule are preserved. The historical balanced reference and the square-root reference are both retained. No iteration/checkpoint switch is allowed to rescue a failed primary comparison.

These small fitting studies are mechanism checks, not independent generalization evidence. A local pass must still lead to a broader study, actual BGRA input/output quality evaluation, a new training seed20261005, and the sealed fresh synthetic draws. Those fresh draws remain unopened. A sufficient minimal model size is still unknown; neither a short fitting failure nor parameter count alone settles that question.

Detailed records: [tone-weight result](tone-balance-fitting-results-20261009.md), [plain-path result](../../jax-plain-path-20261008/docs/plain-path-fitting-results-20261009.md), and [opacity-retention plan](opacity-retention-fitting-plan-20261009.md). The runtime API and Krita plugin are unchanged by these loss-only changes.
