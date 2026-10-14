# DW: boundary-censored geometry results

The candidate does not pass the focused training-quality gate. It improves continuity on some close curves but adds a long duplicate edge to wide example7, endpoint debris to several wide examples, and retains fill-corner holes. Do not expand this unchanged candidate to the reserved confirmation set or adopt it in Krita.

## Controlled comparison

Numerical source commit038f4ec; run `runs/censored-geometry-20261014`. Followed the [frozen plan](censored-geometry-plan-20261014.md): identical DU control initialization,16 fixed training inputs,3,000 updates per arm, unchanged16,064 parameters, six recursion depths and all-stage loss, sqrt(batch)LR. The sole intervention was boundary-censoring distance/derivative supervision outside1 stage pixel. Balanced sign supervision stayed unchanged. Report raw3000, without choosing an intermediate image checkpoint.

The11 preflight tests passed in49.56s. After a style-only lambda-to-function change, the affected test passed again in11.91s; Ruff passed. All3,000 paired prepared inputs matched exactly. All16 cases received every depth101–152 times. Initial functions at six depths, first-six replay metrics/states, restores and16 raw/EMA exports were exact. Training904.732s, evaluation7.778s. Actual parameter changes versus initialization were RMS.0149801/max.1344140 for control and RMS.0137190/max.1065070 for censored. Last300 mean update times were.11725/.12610s. Losses use different objectives and are not cross-arm quality scores.

## Native output review

All384 outputs were generated and all16 predetermined depth6 pages inspected at native resolution. Protected pixels were exactly unchanged.

- Both arms narrow the large input bars. Censored outputs add endpoint/side debris on0–6; example7 develops a second long diagonal edge. Control still has endpoint gaps and uneven thickness.
- Original-source close curves8–11 are generally more continuous under censoring.9/11 still form irregular loops at their fused ends; accepting clean fusion does not require accepting those irregular contours.
- Rough-input curves and crossings retain more pieces with censoring, but long gaps remain. Both fixed-input fits lose robustness compared with initialization on some rough cases. This is consistent with DV's reason to retain rough augmentation in subsequent training.
- Blank outputs stay white. Both arms worsen fill15's corner holes relative to initialization. Censoring reduces some holes compared with control but retains conspicuous lower-left grid-like defects.

Images, prediction hashes and per-page observations are in `evaluation/manifest.json`, `evaluation/metrics.json` and `evaluation/visual-review.json`; `completion.json` verifies the main experiment. `diagnostics/integrity.json` records exposure counts and parameter changes.

## Counterfactual gradient audit

A separately registered protocol (`gradient-plan.json`) compares both objectives on each identical checkpoint/input set, across all six depths and the same six family batches as DV. The script, data, completion and weight hashes are bound in `diagnostics/gradient-conflict/plan.json`; no checkpoint is modified.

At initialization, censoring changes the wide/fill cosine from-.57083 to-.21313 and the blank gradient norm from.68169 to.0002348. This confirms that the unnecessary blank-distance constraint matters to the update. It does not show that removing it solves the visual problem.

At the final censored checkpoint the censored objective has wide/fill cosine+.58023, while the balanced objective at those same weights has-.65313. At the final control checkpoint the balanced objective has+.39698, while the censored objective has-.12587. Thus the sign of this local cosine depends on weights and objective; the negative value in DV was not a persistent impossibility proof. All depth-average cosines remain positive (minimum.7711 across these comparisons). Visual failures persist even when the measured family gradients align.

## Next hypothesis

The sign term averages violations over many already correct subpixels. Small erased holes, short missing segments and extra side fragments can therefore contribute little despite their visual importance. Test a smooth root-mean-square reduction of nonnegative sign-margin violations, keeping correct pixels free, one coherent geometry candidate, and all other geometry/model choices fixed. Use rough augmentation in both matched arms. This is emphasis on sparse white/black condition failures, not a return to teacher-raster MSE. It remains a hypothesis; no claim of improvement is made before training and visual inspection.
