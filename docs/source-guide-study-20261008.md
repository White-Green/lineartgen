# Original rough image as an explicit condition

Loss-only variants, clipping-gradient variants and fixed parameter blends have
not jointly improved halo, line retention and background cleanliness. The next
intervention tests whether exposing information already present before Gaussian
corruption helps the small model. It does not assume information loss is the
only failure cause.

The guide is the original rough source (after synthetic sketch augmentation in
training, or the user's composed rough/lineart image in inference). It is never
the clean target or an upsampled smaller prediction. Downsample both source and
strength before drawing the same independent noise as before. Keep the original
source pyramid fixed through every denoising iteration; recursive predictions
remain a separate clipped state.

Conditioning denoising on a separate source image is established in
[SR3, section2](https://arxiv.org/html/2104.07636v2) and image translation in
[Palette](https://arxiv.org/abs/2111.05826). Our application to a noisy rough
source, small recursive model and partial-edit contract is a new experiment,
not a reproduction or a claim of their reported quality.

Two zero-initialized 3x3 convolutions project the one-channel guide into the
first encoder and first head activations, before their ReLU. Existing convolution
shapes and weights remain intact. The largest model adds36+27=63 parameters:
126,765 ->126,828. Zero additions initially preserve the old function; test the
actual GPU accuracy as well as exact inherited parameter copies. Both branches
must receive finite learning gradients. No capacity sweep is introduced.

Three paired arms add only this conditioning to completed 2k counterparts:

| New arm | Paired counterpart | Primary pixel loss |
|---|---|---|
| source_control | hard control | clipped MSE |
| source_ink_guard | hard ink_guard | clipped MSE |
| source_interval_ink_guard | interval_ink_guard | interval MSE |

The third arm tests whether the previous saturation repair becomes useful when
the original rough information is available. At10k, its improvement is measured
against the completed conventional control and hard ink_guard; there is no
10k interval-only trajectory because that candidate failed its own screen.
Do not describe that final comparison as an isolated conditioning effect.

Keep the common100k EMA, seed20261007, effective batch16, sqrt(batch) peak4e-5,
10k cosine clock, recursion1–6, stage multiplier0, synthetic data/noise/masks and
all quality gates. Diagnostic parameter-gradient comparisons use twelve fixed
training batches, seed20262006 and depths1,1,...,6,6, without coefficient tuning
or model updates. Verify actual six-depth GPU updates/full-state restore, exact
disabled-guide replay of hard and interval baselines, and original input/noise
equality. Unit checks must rule out teacher leakage, guide recomputation from
intermediate outputs, broken mask handling, and an ignored guide.

Model ONNX gains an explicitly declared guide input, while the Python
`LineartModel.infer` API used by Krita remains unchanged. The runtime derives
the guide from its existing input and accepts legacy three-input models too.
Test nonzero guide weights, dynamic ONNX shapes, runtime recursion/repetition
and BGRA output. Keep the installed plugin and distributed model unchanged
while this branch is experimental.

At2k apply the unchanged EMA screen and native-pixel review; continue at most
two passing arms ranked by the sum of primary halo gains. Reuse full optimizer
and EMA states to10k. Final adoption still requires both primary halo gains
>=10%, all line/background safety checks, visual review, an independent
fine-tuning seed and the previously fixed sealed fresh synthetic conditions.
No held-out confirmation is consumed for rejected candidates.
