# AU: fixed learned-field solver convergence

Declared after AT's final review, before running AU. AT failed30/36 toy checks at
32 Euler steps. Hold its final raw6000 bridge model fixed and test32/64/128/256
steps. Zero neural updates, no loss/data/weight changes. This separates evidence
for insufficient integration steps from errors in the learned conditional field.
It cannot prove that all numerical error is eliminated at256.

Use the same12 groups (64px, depths1–3, whole/right-half masks, strengths.1/.5),
same seed2026102088 and64 draws/group. Require all32-step physical/clipped arrays
and metrics to equal AT bitwise/exactly. Inherit its sampler, model and every
numerical file unchanged; bind source hashes before evaluation. Retain all48
output blocks including unclipped final values, and apply AT's36 toy checks
separately at each step count. The final256-step result is primary for this
convergence diagnostic; no new adoption candidate or gate override follows.

Record per-group elapsed inference including initial compilation when present;
these are descriptive, not warmed Krita/ONNX latency benchmarks. Report paired
32→64→128→256 output differences as well as endpoint distance, mode balance,
teacher-averaged MSE, finiteness and protected pixels.32/64/128 are comparisons,
not alternative selected endpoints. Review three native sheets: each of four
mask/strength groups shows teachers A/B, rough, fixed first draw at32/64/128,
and first four draws at256. No cherry-picking. At256 the CNN runs256×depth times.

Existing AT16 tests cover arbitrary positive step counts, oracle integration,
protection, actual updates and state restore; this numerical code is unchanged.
AU additionally verifies actual32-step full-output replay across all12 groups.
This remains a two-shape toy, no new geometry/corpus, independent training seed,
sealed confirmation, model export or plugin/default change.
