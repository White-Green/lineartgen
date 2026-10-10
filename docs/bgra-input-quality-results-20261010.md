# BGRA input contrast — actual-checkpoint diagnostic completed

**Input contrast materially changes faint-line restoration, but scale1.0 alone
does not solve the gray/width/separation problem.** The three frozen models show
the same tradeoff: retaining faint input improves blurred thin lines while also
retaining more of the excessive width in repeated-stroke inputs. No runtime
default, model weights or plugin asset was changed.

The [predeclared diagnostic](bgra-input-quality-plan-20261010.md) ran96 actual
CPU `LineartModel.infer` outputs:16 existing strongly corrupted whole-image cases,
two scales, and E/R/Y. The physical scribble alpha is rounded directly from the
stored rough opacity. Unlike the earlier backend parity test, there is no inverse
contrast correction when constructing BGRA. Target images and layer bytes are
identical between scales. These constructed synthetic layers are not natural
sketches, and the16 development examples are not an independent sample.

Output-BGRA metrics for Y (strength.5):

| Group | Metric | Scale1.1 | Scale1.0 |
| --- | --- | ---: | ---: |
| Corpus blur,4 cases | F1 | .871594 | .901050 |
| | Ink deficit | .190711 | .065189 |
| | Halo | .006799 | .025684 |
| Corpus strokes,4 cases | F1 | .789307 | .768889 |
| | Ink deficit | .094355 | .066064 |
| | Halo | .096266 | .128033 |
| Geometry blur,8 cases | F1 | .286338 | .861946 |
| | Ink deficit | .573916 | .072685 |
| | Halo | .018415 | .042004 |
| | Width MAE,px | .810006 | .602910 |
| | Cuts | .645833 | .000000 |
| | Bridges | .500000 | 1.000000 |

These selected, strong-only, post-BGRA metrics must not be compared directly with
the broader raw-array group means. Full raw/post-BGRA values for all E/R/Y cases
are retained. Geometry raw F1 improves with scale1.0 in E (.253843→.897917),
R (.153293→.462124) and Y (.286338→.861946). Fewer bridges under scale1.1 partly
reflect missing/gray lines, not accurate intentional separation.

A subsequent descriptive comparison against the same16 stored-float development
cases isolates the much smaller input-rounding/backend contribution. Geometry F1
is .898079→.897917 for E, .439394→.462124 for R and .864232→.861946 for Y when
moving from float GPU evaluation to quantized BGRA/CPU with scale1.0. Corpus-group
F1 changes are below.0004. Thus the large scale1.1 thin-line regression is not
explained by ordinary8bit rounding alone in these cases. This comparison was
added after the diagnostic and is retained separately in`quantization-comparison.json`.

Scale1.1 erases41.71%/35.18%/56.08% of nonzero input pixels in the corpus-blur,
corpus-strokes and geometry-blur groups. Remaining total prepared ink mass is
69.09%/76.02%/50.19% of input mass. These figures include faint Gaussian tails
and must not be called the fraction of actual intended strokes removed.
Scale1.0 erases no nonzero input-alpha level during preparation. For example,
curve999 has maximum physical input alpha16/255; scale1.1 reduces its maximum
prepared alpha to.019020 and its total mass from126.992 to9.480. Its output curve
nearly disappears in the reviewed scale1.1 panels.

All three native sheets were viewed. Scale1.0 restores thin curves/corners and
junction branches more clearly in E/Y, while R retains considerable gray. Dense
tire hatching, parallel lines and intended gaps still merge. Repeated-stroke
contours/loops are too thick, often more so at scale1.0. The thick taper remains
dark. This is a quality diagnosis, not a new successful candidate.

All96 JAX/ONNX comparisons satisfy rtol2e-4/atol2e-5 (maximum absolute difference
2.696e-5, covered by the combined tolerance); maximum BGRA difference is1 byte.
Actual infer output equals the separately computed runtime output conversion.
Input/checkpoint/source hashes are unchanged; private exports, predictions,
helper and visual reviews are hash-bound in
`runs/bgra-input-quality-20261010/final-audit.json`.

Future adoption must distinguish teacher cleanup contrast from runtime rough-input
contrast and validate actual layer bytes. The current training augmentation is
applied after teacher normalization, whereas actual BGRA contrast is applied
after rough-image creation. This ordering difference is real; changing only the
deployment scale still requires broad, matched quality checks. Protected-pixel
claims concern prepared float values, not arbitrary original BGRA bytes after
input contrast and output alpha threshold. No Windows/Krita GUI test was run.
