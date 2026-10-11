# BY: compare learned endpoints with the known toy posterior along the path

Zero-update diagnostic registered while BX trains. Compare final raw6000 BV and
BX noisy/source anchors on teacher-interpolated states. This is not free-running
sampling, a quality confirmation or an alternative adoption criterion.
The goal is to locate endpoint errors along time instead of inferring their
cause solely from final rendered shapes. The two known teachers are used only
in the diagnostic reference, never given to the neural predictor.

Use the exact toy teachers/source and Gaussian initialization. At time tau,
each scale observes x=(1-tau)*s+tau*y+(1-tau)*(.25*t)*eps. For equal A/B priors,
sum independent Gaussian log likelihood ratios across active pixels and scales.
With muA/muB and their midpoint m, the log odds are
sum((x-m)*(muA-muB)/variance). The reference endpoint is pA*A+(1-pA)*B.
At tau0 pA=.5 exactly; at later times information arrives through the evolving
state. This posterior uses all scale observations and known prototypes, whereas
the CNN compresses smaller scales through predicted inserts. It is an analytical
reference, not a demonstration that this architecture can attain it.

Depth1/3, whole/right-half masks, strength.1/.5, times
0,.001,.002,.005,.01,.02,.05,.1,.25,.5,.75,.95.64 paths/group with32 A/32 B,
alternating labels and the existing EVAL_SEED/depth/mask/strength noise key.
Same sampled Gaussians for every time and every model within each group.
96 input groups,384 model rows. Final checkpoints only; no selection.
Save each group's input, reference, posterior, four clipped endpoint predictions
and finite/protected checks. Report endpoint RMS versus reference, including
ambiguous-region and common-region errors, posterior entropy, expected
ambiguity and fraction with posterior outside[.1,.9]. No pass/fail quality guard.
RMS uses the existing half-teacher-difference energy normalization.

Tests verify equal priors at tau0, exact Gaussian density-ratio agreement,
independent-scale evidence accumulation, ignored protected pixels and the
near-endpoint teacher limit. Inspect eight native intermediate-state sheets:
depth3, four mask/strength groups, first two fixed paths (A and B), five fixed
times0,.002,.01,.1,.95. Label these as teacher-conditioned states, not sampler
outputs. No new teacher shapes, independent seed, sealed confirmation,
production model, Krita plugin or runtime-default changes.
