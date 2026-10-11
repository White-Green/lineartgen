# BW: finer endpoint integration does not remove mode collapse

The six fixed-weight sampler variants fail 18–22 of the unchanged 36 criteria.
None is a candidate. Four times as many steps reduces some shape error but
preserves severe output-frequency bias and malformed strong-noise shapes.
[Frozen plan](endpoint-time-grid-plan-20261011.md).

| Anchor / integration | Failures /36 | Mean RMS at .1 | Mean RMS at .5 | Fraction within .20 at .5 |
| --- | ---: | ---: | ---: | ---: |
| Noisy / uniform32 | 22 | .166042 | .393049 | .424479 |
| Noisy / sin²32 | 20 | .179508 | .399478 | .419271 |
| Noisy / sin²128 | 20 | .153049 | .363992 | .463542 |
| Source / uniform32 | 18 | .065419 | .560199 | .408854 |
| Source / sin²32 | 18 | .058913 | .536169 | .432292 |
| Source / sin²128 | 18 | .038678 | .484657 | .497396 |

Means combine the six depth/mask groups at each strength. Failure counts are
not quality scores: noisy sin²32 fails fewer guards while average RMS worsens.
Source partial strength .1 still chooses A64/64 at every depth and every grid;
noisy depth2 whole strength .5 still chooses A0/64. More steps do not correct this.

All six native sheets were inspected. White backgrounds remain clean; at .5
several outputs combine a single bar with a double bar along its length, leave
holes or incompletely open the gap. Gray pixels on the protected left half in
partial examples are the original rough source, not restoration halo.
[Noisy depth1](../runs/endpoint-time-grid-20261011/evaluation/noisy-depth-1-native.png),
[depth2](../runs/endpoint-time-grid-20261011/evaluation/noisy-depth-2-native.png),
[depth3](../runs/endpoint-time-grid-20261011/evaluation/noisy-depth-3-native.png);
[source depth1](../runs/endpoint-time-grid-20261011/evaluation/source-depth-1-native.png),
[depth2](../runs/endpoint-time-grid-20261011/evaluation/source-depth-2-native.png),
[depth3](../runs/endpoint-time-grid-20261011/evaluation/source-depth-3-native.png).

Audit: 11 tests passed in9.49s; the actual remaining-time denominator, exact
protection and known-endpoint convergence were tested. All24 BV uniform32
physical/clipped output blocks and metrics replay bitwise. All72 evaluated
blocks are finite with protected pixels exact. 216 guards and six visual hashes
are bound in the [final audit](../runs/endpoint-time-grid-20261011/final-audit.json).
Numerical commit `f144e20`. Zero neural updates; final BV raw6000 weights only.
Sin²128 costs four times uniform32. The sampling denominator changes explicitly
below1/32; BV training and its loss cap are unchanged.

Only the same two memorized synthetic teachers and development Gaussian draws
were tested. No new shapes, independent confirmation, production model, Krita
plugin or runtime-default changes. Further integration-only expansion is not
justified by this comparison.
