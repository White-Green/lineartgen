# Lineart restoration experiment status, 2026-10-09

No replacement has been adopted. The deployed original 100k model remains the baseline. Synthetic rough inputs remain the only corruption source. The required combination—less gray fringe with preserved thin lines, opacity, width, continuity and separation—has not yet passed the complete evaluation.

Recent results and the next controlled comparison:

| Experiment | What it established | Decision |
| --- | --- | --- |
| Full-resolution width16 detail head | Improved local fitting, roughly +.10 blur F1 over the old local guided model; broader results still have gray marks, missing fine strokes, or wrong joins | Broader candidate rejected after native review |
| Progress-conditioned rough/bridge paths with full tone balancing | The learned five-step bridge still forms dark branching structures; blur F1 is .2115/.2488 on train/probe versus .5304/.5804 for the old one-step local reference | Both paths insufficient; bridge advancement gate rejected |
| Tempered tone weights | Square-root weights reduce low-strength blur halos by 47.49%/49.43%, but all 24 strata exceed allowed opacity loss; unit weights fade lines further | Both rejected; all 12 final native sheets inspected |
| Unit-weight rough/bridge paths | Much less false branching, but bridge still loses thin strokes; fails10/18 comparisons after1,200 updates and all10 native sheets reviewed | Declared bridge candidate rejected |
| Square-root weights plus clipped opacity deficit | Coefficients .1/.3 reduce low-noise halos50.03%/51.13% and42.88%/44.91%, but fail19/13 of24 safeguards | Both rejected; all12 final native sheets inspected |
| Physical preclip opacity deficit | Low-noise halo reduction46.09%/49.53% at.1 and42.35%/44.39% at.3; fails16/13 of24 safeguards. Final top-output saturation accounts for only about1–3% of missing opacity in the diagnostic cases | Both rejected;12 required plus2 paired final sheets inspected |
| Fixed single-case fitting | Four independent copies fit one already seen training straight/thin/corner/gap case each, with fixed noise and recursion1–6 | Running in `jax-single-case-fit-20261009`; diagnostic only, never a deployment candidate |

The opacity term measures missing opacity only at editable teacher-line pixels. It has no added force at an already correct gray edge, a protected pixel, or the background. This is a hypothesis about balancing losses, not a guarantee of correct geometry or a proof of the cause of halos. It can still cause excessive darkening under uncertainty; fixed metric safeguards and actual output inspection decide the result.

The completed preclip study started from the original 100k EMA, not a selected fitted endpoint, with the same local 57/57 pairs and random streams. Architecture, recursive depths1–6, stage weighting0, downsample-before-noise, sqrt(batch) learning-rate scaling, and the fixed 1,200-update schedule are preserved. The historical balanced and square-root references are retained, with paired comparisons and six-update replay of both clipped-opacity coefficients. No iteration/checkpoint switch is allowed to rescue a failed primary comparison.

These small fitting studies are mechanism checks, not independent generalization evidence. A local pass must still lead to a broader study, actual BGRA input/output quality evaluation, a new training seed20261005, and the sealed fresh synthetic draws. Those fresh draws remain unopened. A sufficient minimal model size is still unknown; neither a short fitting failure nor parameter count alone settles that question.

Detailed records: [tone-weight result](tone-balance-fitting-results-20261009.md), [plain-path result](../../jax-plain-path-20261008/docs/plain-path-fitting-results-20261009.md), [clipped-opacity result](../../jax-opacity-retention-20261009/docs/opacity-retention-fitting-results-20261009.md), [preclip-opacity result](../../jax-preclip-opacity-20261009/docs/preclip-opacity-fitting-results-20261009.md), and [single-case plan](single-case-fitting-plan-20261009.md). The runtime API and Krita plugin are unchanged by these loss-only changes.
