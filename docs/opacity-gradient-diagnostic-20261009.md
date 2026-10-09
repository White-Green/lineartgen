# Top-output clipping diagnostic

This is a separate diagnostic of the fixed400-update checkpoints, not a change to the active1,200-update experiment or an alternate selection rule. The raw square-root control and opacity coefficients.1/.3 were evaluated on the same first nine probe blur shapes at strengths.1/.5, plus the clean thin example at.5: **57 CPU forwards**, no optimizer updates. Cases and CPU/GPU tolerance were recorded before execution.

The actual CPU clipped outputs match stored GPU images within **7.4804e-6 maximum absolute error**, well below the declared2e-4 audit tolerance. `runs/opacity-gradient-diagnostic-20261009/` contains the plan, source/checkpoint/array hashes, actual preclip arrays, and results. The executed helper and tracked `scripts/diagnose_opacity_clipping.py` have identical source bytes; the helper was added after the experiment's numerical freeze and did not alter any frozen file.

For every teacher-line pixel whose physical preclip prediction exceeded white (+1), the derivative of the opacity term with respect to the clipped image was positive, while its derivative through the hard output clip was exactly zero. The table reports the mean per-case share of missing opacity with that blocked direct derivative, across the nine blur shapes:

| Model at400 | Noise.1 | Noise.5 |
| --- | --- | --- |
| Square-root control | 16.40% | 37.88% |
| Opacity coefficient.1 | 30.43% | 54.22% |
| Opacity coefficient.3 | 14.83% | 48.25% |

At noise.5, the thin/corner/gap examples with coefficient.1 have100% missing opacity and100% of that missing opacity behind the white clip. With coefficient.3, the blocked shares are99.70%/89.85%/97.92%, respectively. The clean thin example has almost none of this blockage (0% for coefficient.3). Moving the white-saturation cutoff outward by2e-4 barely changes the blur aggregates, so the finding is not a near-threshold CPU rounding effect.

This identifies a local gradient route that cannot correct an already saturated white pixel. It does **not** prove that saturation caused the entire training failure, or that the pixel can never recover: other pixels, loss terms, shared weights and recursive features can move its output. More training may also change the result. The existing fixed1,200-update comparison remains authoritative for its own decision.

A subsequent controlled experiment can extend only the one-sided opacity term to the physical preclip value, preserving the existing pixel/background objective and clipped recursive feedback. This differs from replacing every output gradient with STE or replacing the whole pixel objective with interval loss, both of which have already been investigated. It still risks excessive darkening and must pass the same opacity/halo/geometry safeguards.
