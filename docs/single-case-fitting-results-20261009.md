# Fixed single-case fitting and noise transfer

The unchanged model can fit much of an individual cached input, but this is **not a successful restoration model**. At800 updates, three of four depth3 cases pass the numerical fitting screen, while the intended gap remains visibly bridged. Changing only Gaussian noise makes all four fail the same screen. These results favor checking noise diversity and optimization before adding another loss coefficient; they do not prove that capacity is sufficient for the actual task.

The [frozen plan](single-case-fitting-plan-20261009.md) used four independent130,605-parameter copies of preclip-opacity.3 raw at1,200, one already seen training image each, fixed strength.5 noise, batch1 and800 updates. Recursion1–6, stage weighting0, sqrt(batch) LR and other model/loss settings were preserved. The numerical pin is `f35e134a9edc3cb6d1c2520ffd14e61dd7f4e47d` (244 files,7 settings). The CPU-cache preparation failure occurred before any optimizer update and is documented separately in the [preparation record](single-case-fitting-preparation-20261009.md).

| Own case, depth3 | Initial F1 | Final F1 | Final ink deficit | Final halo | Screen |
| --- | ---: | ---: | ---: | ---: | --- |
| Straight | .631371 | .969477 | .000350 | .002801 | Pass |
| Thin | .390597 | .943885 | .000596 | .001788 | Fail F1 |
| Corner | .811097 | .980261 | .001219 | .002142 | Pass |
| Gap | .642978 | .956306 | .002241 | .006511 | Numerical pass; visual gap fails |

The screen remains F1 at50% opacity >=.95, ink deficit <=.05 and halo excess <=.02. Initial/200/800 passage is0/2/3 out of4 at depth3 and0/10/17 out of24 across depths. All six thin conditions fail final F1 (.925247–.943885), as does gap at depth1 (.938356). No final condition fails the opacity or halo threshold. The two required native sheets were actually inspected: straight and corner become much cleaner, thin width/antialiasing remains imperfect, and the deliberate gap stays connected at all shown depths. Global F1 is insufficient to establish local topology.

An additional transfer diagnostic was declared while training was still in progress, before the800-update result. It applies all four specialized models and the common initial model to all four already seen training cases, using both the cached noise and independent noise seed `2026100917 + pair`. This is40 forwards with no updates, not the unopened fresh confirmation.

| Own fitted case, new noise | F1 | Ink deficit | Halo |
| --- | ---: | ---: | ---: |
| Straight | .6093 | .4063 | .0366 |
| Thin | .4725 | .5089 | .0332 |
| Corner | .8959 | .1159 | .0100 |
| Gap | .7504 | .2052 | .0318 |

All four own-case new-noise conditions fail. Other-case transfer is also poor: for example, the corner specialist's thin-line F1 is zero with either noise draw. The native transfer sheet confirms fading and width variation, and the gap specialist still closes the gap. Some central crops look good despite poor whole-image metrics; the160px crop is not the full metric domain. This is strong evidence of case/noise specificity in this fitting diagnostic, not a measurement of generalization to new drawings.

Validation includes four exact initial GPU predictions, finite six-depth preflight updates, exact full-state restore at preflight6 and actual1/200/800, and eight bitwise diagonal comparisons in transfer evaluation. The104 unchanged core/test files match the earlier298-test passing manifest. The new orchestration passed Ruff and Python compilation. Numerical results and hash-bound native reviews are in `runs/single-case-fitting-20261009/` and `runs/single-case-transfer-20261009/`.

Next compare the same four independent fitting problems with fresh Gaussian noise per update, holding cases, initialization, losses and budget fixed. Even a success there would not establish joint multi-image learning, sufficient minimum model size or deployment quality. Production weights, Krita/runtime behavior and the sealed fresh confirmation remain unchanged.
