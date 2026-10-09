# Fresh procedural geometry and combined structural supervision

This is a new generalization/mechanism study, not advancement or adoption of the
four-case connection-only model. Its600-update diagnostic removed the seen gap
but failed8 paired checks and did little for the unseen gap. The fixed2000
endpoint of that experiment remains unchanged and must still be reviewed.

The next question is whether diverse geometry, explicit nearby-paper supervision
and opacity/connectivity retention can jointly avoid the previous tradeoff.
Region loss previously reduced halos while losing thin ink on the old unguided
model; it is reconsidered here alongside the current rough guide, detail head
and preclip opacity retention, rather than claimed as a new loss.

Three models have the same130,605-parameter architecture and start from the same
preclip-opacity.3 raw1200 parent, not a selected four-case specialist:

* pixel: existing sqrt-tone pixel loss and preclip opacity.3;
* region: pixel +.1 times the existing teacher-local line_region_loss;
* structure: region +.1 times the connection-query loss.

The comparison region versus pixel tests local paper/profile pressure under the
current retention loss. Structure versus region tests connection retention and
separation. No hypothesis is accepted from a single loss curve or threshold.

Every update generates four new512px targets with seed2026100951 and independent
update/item keys. Nine kinds cycle evenly: straight, thin, Bezier, corner,
junction, crossing, parallel, gap and taper;5% are blank. Widths.6–6px, random
angles/positions/control points,4x raster supersampling and image scale1.1 match
the existing generator. No development image seeds enter training. Each target
is independently blurred/attenuated by the existing augmentation:80% enabled,
opacity uniform.25–1, sigma uniform0–2. The remaining20% stay clean.
The standard independently sampled Gaussian noise, strength/mask/identity
distribution and downsample-before-noise path remain unchanged. This stage does
not train displaced repeat strokes; those remain a required later evaluation.

Query0 spans20px of a randomly chosen true path. Query1 spans the gap/parallel
separation, or joins two arms of a junction/crossing; otherwise it is inactive.
Off-image anchors are inactive. Patch64,3x3 anchors,eight-neighbor graph, full
V-1/fixed-point search. Actual rasterized teacher capacity, not primitive labels,
decides whether a query is connected. Under partial edits the teacher is the
clean/source composite, and protected pixels get no query gradient. Positive
and negative classes normalize by the number of valid queries separately;
an absent class contributes zero. Queries and targets never enter inference.

Batch4, accumulate1, base LR1e-4 * sqrt(4)=2e-4, warmup100, cosine4000 to1e-5,
EMA.999. Recursive depths cycle1–6, direct stage weight remains0. Final raw4000
is primary, raw1000 intermediate; no endpoint or iteration cherry-picking.

Evaluation uses existing synthetic TRAIN/PROBE records as development only:
the64 four-case/eight-noise conditions and all57 pairs per split, clean/tone/blur,
noise.1/.5, whole/partial, depth3, one iteration. Blank and undefined ink cases
are reported separately, not counted as restoration successes. A follow-up
candidate needs at least10% blur-halo improvement on each split versus pixel,
and per family/split/strength/mask mean ink deficit increase<=.005 and F1
drop<=.005. Whole-image path probes on the four primary cases additionally
require cut/bridge increase<=.01 and width error increase<=.10px. A structure
benefit must also improve gap-middle opacity>=25% versus region on both splits
without breaking those safeguards. These are development screens, not final
adoption. Actual native crops include all nine kinds, both strengths and both
splits, plus protected boundaries and blanks. Full broader real-clean-derived
synthetic rough, repeated inference, BGRA quality, independent training seed
and sealed fresh confirmation are required after a promising candidate.

Validate deterministic data/queries, independent widest-path oracle, correct
gradient sign/class normalization and protected gradients. Freeze source,
settings and evaluation inputs before updates. Six actual GPU preflight updates
cover every recursive depth and full-state restore; actual first six must replay
exactly. At1000 and4000 verify full optimizer/model/EMA restore, fixed model/input
hashes and native reviews. Training a model or reducing one metric does not mark
the user's quality goal complete.
