# R: interval absolute pixel loss, 2026-10-10

Prepare and freeze before inspecting Q's final outputs. Run **only if Q finishes its frozen screen and native review with no passing candidate**. Otherwise prioritize Q confirmation and leave R unrun. Q's numerical code, data, checkpoints and gates remain unchanged.

The fixed O raw4k saturation diagnostic found zero direct full-loss gradient at every incorrect black-saturated pixel in its34 displayed development conditions. In21strokes conditions these pixels account for26.89% of excess-ink mass; in13blur conditions,.97%. This is a selected panel, not a population estimate. The current preclip-opacity term rescues missing ink but gives no correction for excess black. A counterfactual interval absolute pixel term has inward gradients at all those pixels. [Diagnostic](../../jax-source-capacity-20261010/docs/wrong-saturation-gradient-20261010.md).

Add optional `loss.pixel_loss="interval_l1"`. With physical prediction z and teacher y, use distance to the teacher's valid clipping interval: white y=1 allows z>=1, black y=-1 allows z<=-1, interior AA requires z=y. The error is relu(1-z),relu(z+1),or abs(z-y), respectively. Select zero subgradient at exact zero error. Inside[-1,1] the values equal clipped L1; correctly saturated endpoint margins remain allowed. This is an ordinary differentiated loss, not a straight-through clip gradient. Restrict to a single hard-clipped UNet, like existing interval MSE. Existing defaults, clipped L1 and interval MSE remain intact.

`pixel` reuses O `l1` raw/EMA0/1k/4k checkpoints. `interval` changes only this pixel loss and begins at common E raw4k with fresh optimizer/EMA. Keep the small130,605-parameter architecture, source features/source anchor, original-strength one-pass inference and all auxiliary terms, including preclip-opacity.3. The loss changes incorrect white as well as black saturation; it is not a black-only intervention and need not improve topology.

Same O data and schedule: native teachers, original geometry.2,conditional coverage.125,80%mixed rough augmentation,seed2026100961,batch4/accumulation1,4,000 updates,baseLR5e-5×sqrt(4)=1e-4,warmup100,cosine minimum1e-5,EMA.999. Keep sqrt-tone weights, inverse-strength floor.05,identity.1,edge.1,contrast.01,recurrence1–6,stage multiplier0 with recursive gradients and downsample-before-Gaussian. Fixed primary is raw4k; no checkpoint/EMA substitution.

Before training, test values and gradients at wrong/correct saturation, AA exact matches, masks and both strength floors; reject missing preclip and incompatible clip modes. Recheck existing losses, actual interval-L1 optimizer updates and full-state restore. Audit that only pixel_loss differs from O; replay all166 clean hashes, resume6–11, compare complete prepared arrays at all6 depths, and bind expected counts. Freeze code, configs, controls, evaluation inputs and the Q gate source hash. Retain actual startup6 and full states6/1k/4k; save raw/EMA1k/4k. Initial outputs must match O/E exactly because the forward model is unchanged.

Local1k/4k evaluation has64+456 conditions per method (128+912). Broader4k evaluation has E initial, O control and interval over1,566conditions: **4,698 outputs**. Both reused references must reproduce the previous results. Use the same229 checks as Q:

1. Whole geometry blur strength.5: F1>=.90,cut rate<=.02.
2. Whole corpus strength.5 halo: blur<=.02,strokes<=.10.
3. Available real/geometry clean/tone/blur/strokes groups at strength.1/.5,whole/partial: halo<=reference×1.1+.0005,F1 decrease<=.005,ink-deficit increase<=.01 versus both E and O.
4. Whole geometry: width MAE increase<=.10px,cut/bridge increase<=.01 versus both references.
5. Edited black interiors717/727 meanalpha>=.95 (targetalpha>.98,distance>8px); blank maximumalpha<=.05; finite output and protected pixels exact.

Review all8 final local and6 broad sheets. Keep every failure. Any improvement must preserve legitimate AA, thin-line opacity, intended gaps and filled regions; escaping clipping is not itself a quality result. Optional post-training repeats of the already fixed34-condition saturation diagnostic may explain changes but cannot replace the screen. Source-only Gaussian redraws are identical. Inputs remain synthetic and44existing corpus evaluation identities are validation-only. Independent training seed, sealed new conditions and actual BGRA/ONNX validation remain downstream. No production checkpoint, plugin or CI artifact changes.
