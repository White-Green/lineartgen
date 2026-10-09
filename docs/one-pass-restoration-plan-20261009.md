# Explicit one-pass restoration contract

Source-only features ignore intermediate sampled states. The completed broader
diagnostic found N-step predictions equal a single pass at strength/N within
1.2e-6. Add an OPTIONAL sampling_mode=one_pass, restricted to original-source
feature and residual-anchor models. Use the requested strength exactly once per
recursive resolution. Keep the Krita BGRA arguments, strength/masks and1–6-scale
model structure. Valid denoise_steps1–20 remain accepted but are UNUSED by this
mode, as is the added Gaussian seed. This is an intentional model-mode behavior
change, not a claim of identical multi-step behavior. It does not consume previous
predictions to refine them. The ordinary diffusion mode remains the default.

Use ONNX metadata format2 for one_pass so old runtimes reject it instead of
silently reducing strength. The updated runtime supports existing format1 and
new format2. No plugin UI change or production artifact replacement is made.
Models with this metadata need the matching updated runtime.

Before trained-model inference, bind the completed source-signal raw4000
checkpoint and its unchanged130,605 parameters. Change only sampling mode in a
private exported copy. It is a diagnostic candidate with known width/gap/stroke
failures; passing interface tests does not establish adoption quality. The
ongoing opacity-transport experiment remains independent and unchanged.

Use12 fixed existing development cases450,480,510,540,460,490,520,550,1287,1383,
717,727 plus whole/partial white blanks. For each of14 sources use strengths.1/.5
and requested steps1/5/20:84 actual BGRA runtime calls. Construct black-alpha
scribble and gray protected-lineart BGRA from source/strength, decode them using
the actual public input preparation, and use that decoded source for JAX parity.
Source quantization is measured separately, never treated as a model regression.
Teacher is only for review. Additional seed57 calls check seed independence.

Require unchanged parameter tensors, default/one_pass JAX equality for one-step,
ONNX float agreement at rtol2e-4/atol2e-5, at most1 byte BGRA error, exact protected
continuous pixels, and byte-identical outputs across1/5/20 requests and seeds.
Inspect native real/geometry/boundary and blank sheets. The fixed crops come from
the prior broad and teacher-only boundary selections. No training or sealed
evaluation occurs. Report CPU ONNX evidence only, not Windows DirectML/Krita GUI.
