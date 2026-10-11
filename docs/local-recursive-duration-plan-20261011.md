# CJ: continue the matched architectures to6000 updates

CI's15809-parameter local CNN outperformed the134688-parameter U-Net after1200
matched updates, but neither passed the fixed-pair appearance criteria. Continue
both actual model/Adam/EMA states from1200 to6000. This is a duration diagnostic,
not an independent confirmation or a new architecture/loss/data search.

Keep the same batch4, constant LR5e-5*sqrt4=1e-4, no warmup, Adam settings,
noise seed2026101093, four AD512px source/teacher pairs, u=1/.5 rollout with.1 AD
appearance loss, and rotating depths1..6. Use step-indexed inputs1200..5999;
no restart of randomness or optimizer. Keep the original1200-step decay horizon
in metadata: peak and minimum LR are both1e-4, so it remains constant throughout.
Top-stage fitting loss stays an isolated diagnostic. Production per-stage loss
and nested-width models are unchanged; the local pilot has no width nesting.

Test exact inherited state, independent model updates, checkpoint restore and
next-update equality for both architectures. Bind the prior25-test record.
Before SGD reproduce100 fixed CI output blocks; verify full parent state equality,
replay six actual updates and restore complete states1200/1206/3600/6000 with
all16 raw/EMA exports. Record all4800 input hashes and per-update wall times.
No checkpoint selection or early quality-based stopping.

Final raw6000 for both architectures; fixed own initial1200 references. Primary2
and secondary32 inference steps, same source/noise evaluation keys, eight AD
cases, depths1/3/6 and four draws. Each count has384 metric rows,100 output blocks,
352 unchanged fitting checks and192 F1/deficit retention checks for final U-Net
vs own initial, final local vs own initial, and final local vs final U-Net.
Reproduce100 total CI physical/clipped blocks and384 metric rows exactly.
All outputs finite, protected source pixels exact. Review ten fixed sheets:
whole train/probe native/overview with four draws, partial native with one draw
per four training cases. Check hollow thick strokes, weak parallel-line bridges,
mask-boundary dots, width/position error and gray paper, not only F1.

Gap1383 in CI remained open in all four connection probes; its width/position
was wrong. Do not conflate it with the weak bridges between parallel1287 lines.
Continued four-pair fitting cannot establish behavior on diverse widths,
corruptions, strengths, partial masks or ambiguity. Broader synthetic training,
independent seed, sealed conditions, BGRA/ONNX and Krita checks remain later work.
No production, plugin, runtime default, deployment or adoption change.


Execution correction: the first CJ worktree stopped after six updates at a
bitwise replay assertion. Identical inputs and initial full state produced
up to1.49e-8 run-to-run state differences under ordinary GPU settings. Testing
both XLA deterministic flags on this installed ROCm setup produced incorrect
nearest-resize gradients on six of eight exact integer probes (CPU and default
GPU passed all eight), as well as much larger state differences. Those flags
are rejected, not used for training. This is an observation about this installed
setup, not a claim about every JAX/ROCm system. No external bug report is sent.

Keep ordinary XLA flags empty and the original nearest-resize forward operation.
For exact2x upsampling in the local CNN and single U-Net, specify its equivalent
JVP as reshape/broadcast; its transpose sums the four corresponding output
adjoints directly. No parameter, architecture, mathematical loss, data, schedule
or input/output-contract change. Floating-point summation order changes, so the
trajectory need not match ordinary scatter accumulation bitwise. A temporary
six-update diagnostic with this JVP had exact metrics and full model/Adam/EMA
replay. The actual frozen implementation must pass the same check. Verify all100
CI output blocks before learning, plus analytical forward/JVP/VJP tests and the
continuation-state test. Width-nested model implementation remains untouched.
