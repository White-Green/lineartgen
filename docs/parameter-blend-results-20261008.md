# Parameter interpolation: rejected at 10k

The three prespecified interpolations of hard-clipped and straight-through
`neighbor_guard` parameters pass the 2k screen but fail the unchanged final
10k quality gates. No model was adopted. Interpolation exchanges the endpoints'
background cleanliness and faint-line retention; it does not resolve their
joint limitations.

The plan is in [parameter-blend-plan-20261007.md](parameter-blend-plan-20261007.md).
The inference-only study uses the same fixed 1,562 development cases, EMA as the
primary result, raw weights as sensitivity, 55 repeat panels, 40 blank conditions,
and 24 supplemental geometry panels. The two trained endpoints share the 100k
EMA initialization, fine-tuning seed, data draws, 10k schedule, model structure
and losses. Only their clipping derivatives differ. Each blend remains one
126,765-parameter model. No extra training or output ensemble is involved.

`alpha` is the weight of the straight-through endpoint. All three passing 2k
blends were allowed to advance by the plan revision made before any blend
inference and before the 10k endpoint existed. There was no post-result alpha
search. Source commit: `957dd13`.

| alpha | 2k halo gain, real / geometry | 10k halo gain, real / geometry | 10k failed safety comparisons, EMA / raw |
|---|---|---|---|
| .25 | 12.32% / 12.66% | 15.29% / 16.25% | 37 / 35 |
| .50 | 10.44% / 11.39% | 13.81% / 14.91% | 37 / 39 |
| .75 | 8.49% / 10.10% | 12.24% / 13.52% | 42 / 46 |

Halo gains are relative to the paired conventional control, for strong stroke
corruption. The real domain is clean real lineart degraded synthetically, not
human rough drawings. Failed comparisons include overlapping strata and are
not counts of independent defects. Raw .50 and .75 also fail the diagnostic
2k screen on a mild-blur bridge comparison; EMA was fixed as screen primary.

At 10k, .25 and .50 eliminate the straight-through endpoint's background-excess
gate failures, but ink retention and topology still fail. For strong geometry
tone corruption, ink deficit increases by 2.04, 1.81 and 1.58 percentage points
and cuts by 4.17, 3.39 and 3.18 points, respectively. For strong geometry blur,
profile missing fraction changes by +0.23, -1.07 and -2.00 points, but profile
offset error increases by .120, .095 and .073 pixels (allowed increase .05).
Thus lower missing fraction alone is insufficient to establish restored lines.

Eight native-pixel sheets were inspected at each milestone: all 24 selected EMA
geometry crops, eight raw blur crops, real cases 479/480/640 and raw 480, four
one-vs-five-step crops, and a partial blank five-step panel. Strong blur still
erases faint lines. Parallel lines remain merged and deliberate gaps fill in.
Real-image crops retain gray fringes and false marks. Repeated inference can
sharpen these errors without repairing topology. The unchanged 2k screen permits
continuation, not adoption; visual inspection supplies no basis for overriding
the failed 10k gates.

Artifacts are under `runs/parameter-blends-20261007-screen` and
`runs/parameter-blends-20261007-final`: frozen source/settings, sealed endpoints,
construction records, endpoint validation, paired analysis, quality gates,
visual-review manifests, review and completion audits. Each root contains eight
metric files; all defined scalar metrics are finite. Parameter endpoint copies,
export reloads and 36 endpoint forward comparisons were exact on the GPU.
No fresh-condition or second-seed confirmation was consumed for these rejected
models. The shipped Krita model remains unchanged.

The next hypothesis is a proper primary loss on the physical pre-clip output:
distance to the teacher's inverse-clipping interval. Correct saturated white
and black remain stationary; wrong saturation receives a genuine repair
gradient. This differs from passing every auxiliary gradient through clipping.
It needs its own frozen experiment and is not validated by this study.
