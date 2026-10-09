# Secondary reference, declared before the first trained anchor evaluation

The paired primary comparison resets both output projections. To avoid
mistaking recovery from a weaker control initialization for practical progress,
also compare the source arm against the completed procedural pixel model with
its original nonzero denoising heads. Reuse all64 primary and456 broad conditions
at the same1000/4000 steps, with identical dataset and condition keys. Apply the
same frozen quality safeguards. This secondary comparison is observational;
it is not a clean causal ablation of the output anchor because initialization
differs. It cannot rescue failure in the paired primary screen.

The comparison script only combines existing predictions/metrics and invokes
the existing summarizer/renderer. Bind every input artifact by hash; verify
both numerical source manifests and all dataset/condition identities. The
original paired and historical results remain untouched. A candidate should
also beat this established reference before wider confirmation. No fresh sealed
data are generated. The source label in the fourth image column denotes the
predicted source-anchor result; second source column is the rough input.
