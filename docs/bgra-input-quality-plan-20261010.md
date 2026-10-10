# Actual BGRA input contrast diagnostic

Existing development cases bypass BGRA input conversion. The earlier ONNX parity
test divided the stored source by image_scale before encoding it, deliberately
reconstructing the normalized source. That verifies the backend but cancels the
contrast change a physical Krita layer experiences. This diagnostic uses identical
explicit layer bytes with image_scale 1.1 and 1.0, without that inverse correction.

Freeze E initial, R raw4000 and Y raw4000. Export private CPU ONNX copies; E uses
the previously verified one_pass labeling with identical weights. Change only
image_scale in each private runtime instance. No training, model selection,
runtime default or plugin change. Do not use AB intermediate results to choose
models or cases. This is an input-distribution diagnostic, not confirmation.

Use the existing development cases 450/480/510/540 (corpus blur),
460/490/520/550 (corpus strokes), and 807/903/999/1095/1191/1287/1383/1479
(eight geometric blur cases). All have strong synthetic corruption, whole-image
strength .5. Quantize stored normalized rough opacity (1-source)/2 to an explicit
black scribble alpha layer; lineart is transparent. These are newly constructed
synthetic physical layers, not original scanner or human sketch pixels. The
unchanged stored clean image is the desired physical output. No information about
natural sketch prevalence follows from these 16 already seen cases.

Evaluate 16 cases × 2 preprocessing scales × 3 fixed models = 96 outputs using
the actual LineartModel.infer API, seed31, one_pass steps1. Check JAX/ONNX raw
agreement (rtol2e-4,atol2e-5), output BGRA discrepancy at most1 byte, finite output,
and unchanged input/checkpoint/source hashes. Record the number/mass of nonzero
scribble pixels erased by contrast, raw and output-BGRA F1/ink deficit/halo,
geometry widths/cuts/bridges. Report all cases and group means; do not invent an
adoption pass from this small diagnostic. Review three native crop sheets with
target/source and both scales for E/R/Y. Bound helper, plan, sources, input bytes,
models, exports, predictions and reviews by hashes in a separate ignored run.

This whole-image diagnostic makes no claim about arbitrary protected BGRA bytes:
prepared float pixels are preserved at strength0, while input contrast and output
alpha threshold can change the representation before/after the model.
