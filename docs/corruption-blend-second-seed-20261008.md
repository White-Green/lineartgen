# Independent fine-tuning seed for the selected 1/16 blend

The selected development recipe is fixed: combine the EMA of guided strokes and
guided mixed-corruption ink guard as 15/16 and 1/16 after exactly 2,000 updates.
Use seed20261006 for both endpoint runs, keep the original 10k cosine schedule,
and retain the same 100k initialization. This tests fine-tuning randomness, not
an independent original training seed. No alpha, loss coefficient, corruption
range or gate is retuned.

Coordinator: `runs/corruption-blend-second-seed-20261008`. Endpoints:
`runs/corruption-seed20261006-strokes` and `runs/corruption-seed20261006-mixed`.
Separate existing trainers ensure each recipe gets its own correct corruption
while consuming the same indexed clean draws, geometry and Gaussian keys. Only
the selected blend and matched control need complete quality evaluation; endpoint
full states/exports and interpolation endpoint equivalence are separately audited.

The existing seed20261006 control at 2k can be reused. Its numerical source
archive and settings were verified against the original source manifest (the
historical checkout has since advanced). Its config differs from source_control
only in source-conditioning and fine-tuning seed. Dataset and evaluation inputs,
including every manifest/file digest, match exactly; only an explanatory `role`
label was changed in later metadata. The first setup assertion caught that text
difference before any training; its incomplete folder and failure log were kept.
The retry explicitly ignores only that label and records both labels.

Actual GPU six-depth control updates under the current code reproduce the old
control's complete model, optimizer and EMA arrays exactly. Both new endpoint
recipes also receive six-depth GPU updates and exact full-state restoration
before their real run. Training/metric/runtime code and tests are byte-identical
to the preceding 245-test suite; this suite is inherited with hashes and explicit
provenance. New preparation/orchestration is exercised with actual sealed inputs
and checkpoint restoration rather than claiming a new full-suite run.

After training, construct only the fixed 1/16 EMA/raw models. Re-evaluate the old
2k control with current inference code and require exact agreement with its
previous 1,562-case metrics, then apply all unchanged strict final safeguards
to the selected blend. Review native crops, partial masks, repeat inference and
blank images. Do not generate the prospective fresh synthetic set unless this
seed comparison also passes. Fresh selection seals must bind both chosen models,
their controls and the evaluation implementation before generating inputs.

The first training launch stopped before its first optimizer update because the
protocol lacked `prior_ramp_steps`, which the shared trainer reads even when the
critic coefficient is zero. Restore the original value (1000) from each recipe,
archive the original source/protocol manifests, and reseal before restarting.
No loss, data or optimizer state is changed by this metadata repair. A real
one-update smoke run is retained and resumed as part of the same 2k trajectory.
