# Structural supervision with original-rough CNN features

The source-signal1000 screen improves blur F1, opacity and halo together, and
native images have stable fine lines/corners. Intended gaps still become joined,
and widths/tips remain imperfect. Test region and connection supervision with
this easier feature input. Earlier identical loss coefficients on noisy features
failed; this is an interaction/generalization study, not a claim those losses
are new or that source-signal is already accepted. Its fixed4000 endpoint and
review continue unchanged.

Control pixel is the source-signal arm (original rough features and source output
anchor). New region is base +.1 line_region_loss. New structure is region +.1
connection-query loss. Use exactly the existing procedural objective code,
queries and coefficients. Actual teacher maximin capacity defines whether a
query is connected; the teacher is never a model input. Predicted graph-path
gradients remain sparse, so this does not guarantee global topology.

All three start from the same zero-output-head initial weights as the source
anchor and source-signal studies, NOT a selected trained source-signal endpoint.
130,605 parameters; features use original rough images, noise map, recursive
smaller prediction and rough guide. Output source +sqrt(t)F, exact protected
incoming image at t=0. Keep noise generation after downsampling, depths1–6,
stage multiplier0, preclip opacity-deficit.3, sqrt-tone pixel loss and existing
identity/edge/contrast terms. No transport loss or residual-floor change.

Same procedural seed2026100951,4 new512px examples/update, nine kinds and5% blank,
80% blur/opacity augmentation, sigma0–2/opacity.25–1, batch4, baseLR1e-4*sqrt(4),
warmup100, cosine4000 to1e-5, EMA.999. Primary raw4000; intermediate1000. Train
only the two new arms, rotating their update order. Copy the completed control's
0/1000/4000 snapshots exactly, independently replay its first6 full updates and
require identical64 primary and456 broad outputs/metrics. The treatment
preflight/actual startup includes identical step0 evaluation; require exact
six-update replay, full-state restore and raw/EMA checkpoint agreement.

Use the unchanged64 primary and456 broad conditions per model. >=10% aggregate
blur-halo reduction on each split; per family/split/strength/mask ink-deficit
increase<=.005 and F1 decrease<=.005; per primary case width-error increase<=.10px
and cut/bridge increase<=.01. Structure must also reduce gap-middle opacity>=25%
versus region in both splits without violating safeguards. Inspect all8 final
native sheets: nine shapes at both strengths/splits, blank and protected edges.
All three ignore added Gaussian seed; the eight primary draw repetitions are
therefore NOT eight independent outputs, and should not be treated as evidence
of eight independent generalization successes.

Freeze code/settings/inputs before updates. This is a development comparison,
not sealed confirmation. A relative win must also have satisfactory absolute
native line quality and beat established references before broader synthetic
rough, repeated inference/BGRA, independent seed and sealed draws. No production
replacement or Krita GUI claim. Do not choose an intermediate endpoint to
rescue failure at4000.
