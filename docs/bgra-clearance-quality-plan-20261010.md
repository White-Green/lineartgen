# Fixed AB final checkpoint at the BGRA boundary

After the scheduled AB raw4000 update, but before its broad development results,
append the fixed checkpoint to the completed BGRA input diagnostic. Reuse its
same16 cases, exact physical scribble bytes, clean targets, scales1.1/1.0,
strength.5, seed31 and one_pass steps1. No intermediate/EMA selection, new source
identity, model fitting or deployment change. This adds32 actual CPU ONNX outputs;
E/R/Y's existing96 remain descriptive references, not fresh runs.

Use the same JAX/ONNX float and BGRA parity checks, raw/post-BGRA quality metrics,
input and numerical source bindings. Verify every input hash against the prior
diagnostic and bind its metrics/predictions. Review three native sheets showing
target/source, Y at both scales and AB at both scales. This is a development
quality diagnostic, not independent confirmation or a runtime-default decision.
