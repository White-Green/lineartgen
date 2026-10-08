# Teacher tone weights and uncertain line positions

The current tone-balanced pixel loss can favor a broad dark prediction when a
line's position is uncertain. An analytic toy example verifies this property
using the actual loss implementation. It does not establish that the observed
neural halos are caused by tone weighting, and changes no active experiment.

For a binary teacher with ink fraction `q`, ignoring the tiny normalization
epsilon, the black-to-white weight ratio is `(1-q)/q`. If the conditional
probability of ink at a pixel is `p`, the unconstrained weighted-MSE optimum in
ink units is

```
z = p*(1-q) / [p*(1-q) + (1-p)*q].
```

Plain MSE instead gives `z=p`. Thus balanced output reaches .5 already at `p=q`,
rather than `p=.5`. This rewards retaining rare ink, but the output is not a
calibrated ink probability. For uncertain positions it may occupy much more area
than any individual teacher. Neither pixelwise conditional mean guarantees a
coherent, thin line drawing.

In the existing local fitting corpus, the36 nonblank distinct geometric teachers
across train/probe have mean ink fractions .000450–.007475. Their ideal binary
black-to-white ratios would be about133–2,221. These teachers are antialiased,
so the binary ratio is illustrative; the actual weights vary continuously with
their pixel opacity. This is a corpus diagnostic, not measured network uncertainty.

The separately declared toy ensemble contains21 two-pixel-wide black vertical
lines in512px images, displaced from−10 to+10 with a normalized Gaussian position
distribution of sigma2. Each line spans256 rows, giving `q=1/512`. All positions
are deliberately treated as indistinguishable inputs. This hypothetical ambiguity
is not a claim that the stored float blur pairs are indistinguishable; the known
blur inverse diagnostic shows that those retain recoverable information.

Compute the exact weighted optimum from the actual `tone_weights` arrays, then
differentiate the actual pixel loss at that optimum with strength.1. Edge and
contrast terms are disabled only in this diagnostic, to isolate the pixel term.
Maximum absolute gradients are2.21e-12 (plain) and9.80e-11 (balanced), confirming
the numerical stationary points of the convex unconstrained objectives.

| Analytic output | Peak ink | Width above alpha.1 | Width above alpha.5 | Cross-section ink sum |
| --- | ---: | ---: | ---: | ---: |
| Each teacher |1|2px|2px|2|
| Plain-MSE optimum |.3755|6px|0px|2|
| Tone-balanced optimum |.9968|16px|14px|13.174|

The native sheet was actually inspected: teacher examples are thin black lines;
the plain optimum is a faint gray ridge; the balanced optimum is a wide black
stripe with gray sides. It contains no learned-model predictions. The two loss
values use different objectives and are not a quality ranking.

This identifies a loss tradeoff worth isolating if further experiments are needed.
Simply disabling balance can lose uncertain thin lines, as the plain optimum
also illustrates. A matched training comparison, retention/topology checks and
actual output review are necessary before choosing unbalanced or tempered
weights. The pending conditional-path probe intentionally keeps the current loss
in both arms so its path comparison is not confounded by such a change.

All data, prior plan, as-run script, numerical checks and actual review record are
under `runs/tone-balance-diagnostic-20261008/`. The tracked
`scripts/diagnose_tone_balance.py --root NEW_ROOT` expects a plan and refuses
overwriting results. Its replay reproduces every numerical result and the native
sheet exactly. No candidate training, production change or new confirmation draw
was performed by this diagnostic.
