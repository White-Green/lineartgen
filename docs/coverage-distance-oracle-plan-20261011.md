# BG: coverage-preserving distance extension, before learning

The current velocity experiments fail even fixed geometry. Investigate richer
geometry supervision separately from further time/LR sweeps. Distance transforms
can support differentiable rasterization ([Mihai and Hare](https://arxiv.org/abs/2103.16194));
DeepLSD predicts an attraction field for straight-line detection
([Pautrat et al.](https://arxiv.org/abs/2212.07766)). These motivate an investigation,
not a claim that their tasks or methods solve this restoration problem. No code
or model is imported from those works.

Zero learned updates. Compare a thresholded-mask distance field against a
coverage-preserving scalar field on exactly the eight AD512px clean teachers,
plus paper, black, faint line(alpha.2), half-coverage line(alpha.5), one-pixel gap,
and a tile containing all256 alpha values. These14 targets include deliberately
non-lineart images: invertibility is not proof of a lineart prior. No new sealed
or final data. Radius8 pixels fixed.

For the coverage field d: fractional clean y in(-1,1) maps to y/2. Saturated paper
maps to min(EDT(paper)-.5,8); saturated ink maps to -min(EDT(ink)-.5,8). EDT measures
pixel-center distance to the nearest nonmember. Uniform paper/ink are explicitly
+8/-8 to avoid a fictitious external image border. Render y=clip(2*d,-1,1).
This reproduces the original fractional coverage by construction, while providing
spatially extended targets outside the antialias band. It is NOT an exact
continuous SDF, Eikonal field, topology guarantee, width prior or new information
at fractional pixels. Do not impose unit-gradient regularization based on this
construction. No change to the network or its physical integration field yet.

The threshold comparator uses mask(y<=0), analogous half-pixel-offset positive
and negative EDT, and the same renderer. Test all14 full fields and additionally
render each under a left-half-protected rough source. Require max absolute error
<=1e-6 for coverage round trips, exact protected prepared-float pixels and finite
bounded outputs. Compute ordinary halo, deficit, alpha50 F1 and coverage overlap;
AD target probes/fit criteria verify no gap or thin-line distortion. Metrics for
binary reconstruction are diagnostic, not pass overrides. Save28 field/output
blocks and three sheets: eight AD native crops, six extra fixed native crops,
and coverage-field visualization. Teacher-only oracle cannot establish quality
of a learned model. A future distance-supervision loss needs its own frozen plan.
