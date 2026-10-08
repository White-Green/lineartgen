# Broader detail-head comparison

Declared after the completed local fitting study and before any broader candidate
update or inference. Only the width16 full-resolution branch advances. All new
trajectories restart from the original common100k EMA, with zero added outputs;
they do not reuse the local57-pair fitting checkpoint.

Train two independent130,605-parameter models: `detail_strokes` and`detail_mixed`.
Their counterparts are the completed raw-guide `source_control` and
`mixed_source_control`. Each candidate differs from its counterpart only by
`model.detail_channels=16`. Use conventional clipped MSE and the unchanged
identity.1/edge.1/contrast.01 terms. The strokes data retain50% clean inputs and
50% synthetic displaced/widened strokes. Mixed data retain50% clean and use equal
shares of tone/blur/strokes among the corrupted half. Existing ranges, geometry
mix, blanks and dataset splits stay fixed. Clean real art is permitted only as
input to synthetic corruption; no natural rough/clean pairs are used.

Keep training seed20261007, common100k initialization, optimizer/EMA reset,
batch16 as4×4 accumulation, baseLR1e-5 with sqrt(batch) scaling, peak4e-5,
minimum1e-6, warmup100 and the original10k cosine clock. Keep recursion1–6,
stage weights0, downsample-before-noise, independent Gaussian noise and masks.
No parameter blends, guide-encoding changes, new loss coefficients, residual
floor or input-scale changes are combined with this comparison.

Before training, require inherited270-test coverage with every tested numerical
source byte unchanged, candidate GPU updates at all6 depths with exact full-state
restore, and paired six-update historical control replay. Audit zero-added-output
GPU inference at1–6 levels and1/2/5 sampling steps. Preserve source/settings and
all checkpoints. Restore the first actual update before continuing.

At2k, evaluate the inherited1,562 development cases, EMA primary/raw diagnostic,
repeat and blank panels and fixed native crops. Apply the existing2k screen
(halo reduction at least5% in one primary domain, retention/topology safeguards)
and native review. Compare both the unconditioned campaign control and each
matched guided counterpart. A candidate must show actual visual support; lower
loss or the local fitting result alone cannot advance it.

Up to two passing candidates may continue the exact full state to10k under the
unchanged clock. Preserve or reconstruct each guided counterpart's exact2k full
state for a matched continuation if its completed10k counterpart is unavailable.
No reset, best-step selection, longer extension, alpha sweep or EMA/raw switch.
At10k apply the existing STRICT FINAL campaign gates, including at least10% halo
reduction in both domains and all thin-line/cut/bridge/width/offset/background/F1
limits. Inspect actual synthetic BGRA quality and measure inference latency for
any surviving architecture. Only a fully passing candidate can proceed to new
training seed20261005 with a newly trained matched control and then sealed new
synthetic draws. Those draws remain unopened. Windows DirectML/Krita GUI quality
cannot be inferred from CPU ONNX parity.

The local study leaves missing thin lines and false connections; adoption is not
implied. It also gives only a convolution-MAC estimate (+42.1% for width16), not
measured runtime. The [input diagnostic](input-recoverability-diagnostic-20261008.md)
is separate: its known blur metadata is unavailable for arbitrary user inputs,
and it does not alter these acceptance rules or the runtime image scale1.1.
