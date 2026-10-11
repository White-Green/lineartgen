# BF: fresh initialization for the512px geometry flow

BC/BE inherited a narrow two-vertical-bar toy prior. They do not show that the
same velocity model initialized afresh cannot learn the four antialiased shapes.
Isolate initialization before changing model size or geometry representation.

Train one fresh130628-parameter model(seed42), with exactly BC's configuration,
prepared_batch implementation, data, noise, times, ordering, optimizer, loss and
6,000 updates. Channels4/10/20/40 detail16, source/progress conditioning, noisy
features and residual base remain unchanged. Fresh Adam and EMA=model. Batch4,
baseLR5e-5 times sqrt(4)=constant1e-4, warmup0, seed2026101093, cycle depths1–6,
highest-stage interval velocity loss with gradients through smaller predicted
endpoints. Noise after resize,25%tau0 else sin²(pi*U/2). No time-focus change,
projection, target shuffling or added geometric loss. Compare against frozen BC
interval6000, whose initial weights were the learned toy. This is not equal
total lifetime compute: the frozen control includes prior toy pretraining.

Retain fresh states0/6/2000/6000 and8 raw/EMA exports. First six updates replay
exactly; full restores and export equality required. Final raw6000 only. Same
BC original32-step grid sampler, cases, sources, crops, evaluation seed, four
draws and depths1/3/6. Reproduce all25 frozen-control final full/partial blocks
bitwise. Evaluate200 rows:96 fresh,96 control,8 fresh-initial references. Keep52
blocks (48 final full,2 initial full,2 partial). Primary88 fit checks per arm and
64 fresh-vs-control F1/deficit retention checks keep BC thresholds. Unfitted probes
are not trained or used to choose checkpoints. All other depths diagnostic.

Inspect five sheets with every primary draw: train/probe native and overview,
plus native partial mask. Require finite outputs and exact protected prepared
floats. Partial-mask output remains an invariant check. A fixed-pair success is
not generalization, an independent training seed, sealed broad confirmation or
actualBGRA/Krita quality. No production/plugin/default changes.
