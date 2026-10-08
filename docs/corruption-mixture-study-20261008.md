# Prospective corruption-family mixture study

The source-guide 10k study reduced gray halo by 25–29% with ink guard but still
lost faint tone-only lines and increased cuts on mild corruption. It explicitly
trained displaced/widened/blurred strokes, but never reserved samples for
opacity-only or blur-only input. The hypothesis is a coverage gap in synthetic
training, particularly now that the model can directly see the rough guide.

Before any candidate update, freeze two arms: `mixed_source_control` and
`mixed_source_ink_guard`. Each differs from its completed guided counterpart
only in `sketch.mode`: `strokes` → `mixed`. Total augmentation probability stays
0.5. On enabled examples choose tone, blur or strokes uniformly and independently;
expected total fractions are 1/2 clean and 1/6 each family. Reuse existing opacity,
sigma, displacement and dilation ranges, without tuning to development images.

Start from the common original 100k EMA with zero guide projections, as before.
Do not continue the rejected 10k models. Keep the architecture, losses and their
coefficients, training seed, clean draws, Gaussian keys, masks, recursion 1–6,
stage weights, sqrt(batch) LR and 10k cosine clock unchanged. The intervention
necessarily changes rough inputs and protected-region targets; clean images and
Gaussian draws remain paired. No model size or inference/Krita API change.

Old modes must be bitwise unchanged. Test mixture equivalence to individually
chosen legacy families, clean/white preservation and downsample-before-noise.
Replay both original guided baselines through six GPU updates and compare their
complete model/optimizer/EMA arrays with recorded preflight states. Exercise
both new arms at depths 1–6 and restore full state. Inspect family allocations
on twelve training-only batches; no gradient or coefficient fitting is needed
for this data-only change.

Use the existing 1,562 development cases and fixed EMA 2k screen, native visual
review, raw sensitivity, repeated inference and blank diagnostics. Continue at
most two passing arms in summed primary-halo order with their exact optimizer
and EMA clocks. Compare to both conventional hard baselines and the corresponding
guided-strokes baseline at 2k and 10k. Only the latter isolates this intervention.
No gate changes: final adoption still needs at least 10% halo reduction in both
primary domains plus all retention safeguards, native output review, a second
fine-tuning seed and sealed fresh synthetic conditions. Confirmations remain
unopened until an eligible candidate exists.
