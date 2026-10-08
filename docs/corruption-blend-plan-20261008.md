# Fixed small corruption-parameter blends, prospective plan

The guided strokes-trained ink guard reduces halo but slightly loses faint tone
lines; the mixed-corruption counterpart greatly improves faint-line retention
but increases halo, width and bridges. Their common 100k initialization and 2k
fine-tuning clock align parameters. Test whether a small weight interpolation
can retain the first model's denoising and some of the second's retention.
No improvement is assumed from linear parameter interpolation.

Before constructing or running any blend, fix alpha to 1/16, 1/8 and 1/4:
`theta = (1-alpha)*source_ink_guard + alpha*mixed_source_ink_guard`.
Use the same alpha for EMA and raw. Seal both endpoints' weights, metadata and
configs and the existing acceptance/confirmation plans. Keep one model with
126,828 parameters; there is no output ensemble and no new training/optimizer.

This is a new inference-only study at the fixed existing 2k checkpoints. Apply
the unchanged **strict final** numerical criteria at once, not the permissive
2k training screen. Matched conventional control remains the primary baseline;
also report both endpoints, hard ink guard and common initial model. The earlier
mixture screening rejection and guided 10k rejection remain valid. This plan
does not authorize further training of either rejected endpoint.

Verify alpha0/1 exact parameter and GPU forward equivalence, including actual
rough guides at 1/2/5 inference steps; verify save/reload exactness. Run all
1,562 development cases, EMA/raw, fixed native image review, repeats, blanks and
supplemental shapes. Rank strict-final and visually acceptable EMA candidates by
summed primary halo gains, select at most one, and do not change alpha or gates
after viewing blend results. Raw is a sensitivity check, never a replacement.

A selected model must be reproduced with exactly the same alpha after separately
fine-tuning both endpoints to 2k with seed20261006 and paired control. Only after
that comparison may its checkpoint hashes be sealed for the previously declared
fresh synthetic conditions. No fresh inputs have been generated. If no blend
qualifies, record rejection and change the hypothesis; do not extend the alpha
grid based on this result.
