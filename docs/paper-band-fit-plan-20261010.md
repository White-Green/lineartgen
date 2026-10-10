# AD: near-line paper-band loss fitting diagnostic

AB's gap-clearance loss partially improves broad halo/separation. AC's extra
general geometry does not improve blurred-geometry separation. AA's matched
four-pair fitting still has excessive corner/parallel halo and curve F1 below.98.
Closing-derived gap importance is zero around isolated straight contours, so it
does not directly emphasize the ordinary gray fringe beside a line.

Test one additional teacher-only term on the same fixed fitting task before
another broad run. Let alpha=stop_gradient(clip((1-target)/2,0,1)) and d=max_pool
alpha with radius3. On active pixels with alpha<=.02 and d>.25, weight=relu(d-alpha),
otherwise0. The loss is per-image normalized weighted relu((1-physical_preclip)/2-alpha).
Add coefficient.1. It acts on near-line paper, including some already weighted
gaps; the existing gap loss retains additional emphasis there. It does not directly
penalize teacher ink above.02 or protected pixels, but shared parameters may still
reduce real ink. The radius is square, not a Euclidean or topology guarantee.

One `band` arm: same130,605-parameter model,R raw4k initialization,fresh optimizer/
EMA and4 fixed pairs as AA. Keep gap weight.1/radius3 and all other AA losses;
only paper_band_weight0→.1 changes. Reuse AA `gap` raw0/100/400/1200 controls,
requiring all8 predictions/metrics at each milestone to reproduce exactly.

Train999curve/1095corner/1287parallel/1383gap, descriptive probes807straight/
903thin/1191junction/1479taper. All are existing development figures at512px with
strong blur,strength.5 and whole-image editing. Repeat the exact pairs with no
new augmentation. Cycle depths1–6 evenly for1200 updates,batch4,accumulation1,
baseLR5e-5×sqrt4=1e-4 constant,min1e-4,warm0; stage multiplier0 with recursive
gradients,downsample-before-noise,clipping/decay/EMA unchanged. No AA fitted
weights,intermediate/EMA selection or candidate adoption.

Before freeze, audit the teacher mask and physical-output gradient on the four
initial examples. Require exact-target zero loss/gradient, no direct ink/protected
gradient, inward correction for false black saturation, empty/uniform fields and
rotation symmetry. Defaults must exactly match AA's loss/gradient. Test real
update/full-state/next-update restoration and generic training preclip routes.

Evaluate0/100/400/1200 and review both final native sheets. Keep the22 per-model
fitting criteria: each training case F1>=.98,deficit<=.02,halo<=.003,widthMAE<=.2,
cuts/bridges0 where valid. Also require no trained-case F1 decrease>.005 or deficit
increase>.01 versus AA before considering a broad follow-up. Unfitted probes are
descriptive and must be reported, especially thick taper damage. Initial output
checks at all6 depths,actual6/100/400/1200 states and6 raw/EMA exports are retained.

This is a loss mechanism/fitting diagnostic on development figures. Its weights
must never be adopted. No corpus/sealed/final-source evaluation,independent seed,
production/plugin changes or runtime input-scale change in this study.
